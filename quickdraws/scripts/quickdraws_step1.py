# This file is part of the Quickdraws GWAS software suite.
#
# Copyright (C) 2024 Quickdraws Developers
#
# Quickdraws is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Quickdraws is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Quickdraws. If not, see <http://www.gnu.org/licenses/>.


import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
import time
import h5py
import logging
from datetime import datetime
import warnings
from pathlib import Path
import pandas as pd 
from pysnptools.snpreader import Bed

import quickdraws.scripts
from quickdraws import (
    preprocess_phenotypes,
    PreparePhenoRHE,
    runRHE,
    MakeAnnotation,
    convert_to_hdf5,
    blr_spike_slab,
    str_to_bool
)

from quickdraws.scripts import get_copyright_string


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def main():
            
    overall_st = time.time()
    ######      Setting the random seeds         ######
    # Set base random seeds consistently across all processes
    # This ensures reproducible initialization while allowing
    # DDP to naturally diverge through different data partitions
    base_seed = 2
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(base_seed)

    ######      Parsing the input arguments       ######
    parser = argparse.ArgumentParser()
    parser.add_argument("--bed", "-g", help="prefix for bed/bim/fam files", type=str)
    parser.add_argument(
        "--covarFile",
        "-c",
        help='file with covariates; should be in "FID,IID,Var1,Var2,..." format and tsv',
        type=str,
    )
    parser.add_argument(
        "--phenoFile",
        "-p",
        help='phenotype file; should be in "FID,IID,Trait" format and tsv',
        type=str,
    )
    parser.add_argument(
        "--keepFile",
        "-r",
        help='file with sample id to keep; should be in "FID,IID" format and tsv',
        type=str,
    )
    parser.add_argument(
        "--out",
        "-o",
        help="prefix for where to save any results or files",
        default="out",
    )
    parser.add_argument(
        "--annot",
        help="file with annotation; one column per component; no overlapping",
        type=str,
    )
    parser.add_argument(
        "--kinship",
        help="King table file which stores relative information of upto 3rd degree relatives (tab-seperated and has ID1 ID2 Kinship as columns)",
        type=str,
    )
    parser.add_argument(
        "--ldscores",
        help="Path to ldscores file (should have MAF and LDSCORE columns and tab-seperated)",
        type=str,
        default=None
    )
    parser.add_argument(
        "--modelSnps",
        help="Path to list of SNPs to be considered in model fitting",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--rhemc",
        type=str,
        default=str(Path(os.path.dirname(os.path.abspath(__file__)),"GENIE_multi_pheno")),
        help="path to RHE-MCMT binary file",
    )
    parser.add_argument("--out_step0", help="prefix of the output files from step 0", type=str) ## depreciate
    parser.add_argument("--hdf5", help="master hdf5 file which stores genotype matrix in binary format", type=str)
    parser.add_argument("--h2_file", type=str, help="File containing estimated h2")
    parser.add_argument(
        "--h2_grid",
        help="grid search for h2 instead",
        action="store_const",
        const=True,
        default=False,
    )
    ## hyperparameters arguments
    parser.add_argument(
        "--num_epochs", help="number of epochs to train loco run", type=int, default=40
    )
    parser.add_argument(
        "--alpha_search_epochs", help="number of epochs to train for alpha search", type=int, default=80
    )
    parser.add_argument(
        "--validate_every", help="How often do you wanna validate the whole genomre regression (default = -1, which means never)", type=int, default=-1
    )
    parser.add_argument(
        "--early_stopping_patience", help="Number of epochs to wait for improvement before stopping each model (0 disables early stopping)", type=int, default=10
    )
    parser.add_argument(
        "--early_stopping_min_delta", help="Minimum improvement in validation loss to reset patience counter", type=float, default=1e-4
    )
    parser.add_argument(
        "--lr",
        help="Learning rate of the optimizer",
        type=float,
        nargs="+",
        default=[
            4e-4,
            2e-4,
            2e-4,
            1e-4,
            2e-5,
            5e-6,
        ],
    )
    parser.add_argument(
        "--alpha",
        help="Sparsity grid for Bayesian linear regression",
        type=float,
        nargs="+",
        default=[
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.5,
        ],
    )
    parser.add_argument(
        "-scheduler",
        "--cosine_scheduler",
        help="Cosine scheduling the outer learning rate",
        type=str_to_bool,
        default="false",
    )
    parser.add_argument(
        "--batch_size", help="Batch size of the dataloader per GPU (will be scaled automatically for DDP)", type=int, default=128
    )
    parser.add_argument(
        "--forward_passes", help="Number of forward passes in blr", type=int, default=1
    )
    parser.add_argument(
        "--num_workers",
        help="torch.utils.data.DataLoader num_workers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--train_split",
        help="The training split proportion in (0,1)",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--binary",
        help="Is the phenotype binary ?",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--lowmem",
        help="Enable low memory version",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--rhe_random_vectors",
        help="Number of random vectors in RHE MC",
        type=int,
        default=50
    )
    parser.add_argument(
        "--rhe_jn",
        help="Number of jack-knife partitions in RHE MC",
        type=int,
        default=10
    )
    parser.add_argument(
        "--phen_thres",
        help="The phenotyping rate threshold below which the phenotype isn't used to perform GWAS",
        type=float,
        default=0
    )
    parser.add_argument(
        "--predBetasFlag",
        help="Indicate if you want to calculate and store the prediction posterior betas",
        action="store_const",
        const=True,
        default=False,
    )
    parser.add_argument("--chunksize", help="Chunk size of the HDF5 file, higher usually leads to faster conversion but might require more RAM, should be divisible by 128", type=int, default=8192)
    
    ## DDP arguments
    ddp_group = parser.add_argument_group("Distributed Data Parallel")
    ddp_group.add_argument(
        "--ddp",
        help="Enable Distributed Data Parallel training",
        action="store_true",
        default=False,
    )
    ddp_group.add_argument(
        "--local_rank",
        help="Local rank for DDP (automatically set by torchrun)",
        type=int,
        default=-1,
    )
    ddp_group.add_argument(
        "--world_size",
        help="Total number of processes for DDP",
        type=int,
        default=1,
    )
    ddp_group.add_argument(
        "--master_addr",
        help="Master address for DDP",
        type=str,
        default="localhost",
    )
    ddp_group.add_argument(
        "--master_port",
        help="Master port for DDP",
        type=str,
        default="12355",
    )
    
    ## wandb arguments
    wandb_group = parser.add_argument_group("WandB")
    wandb_mode = wandb_group.add_mutually_exclusive_group()
    wandb_mode.add_argument(
        "--wandb_mode",
        default="disabled",
        help="mode for wandb logging, useful while debugging",
    )
    wandb_group.add_argument(
        "--wandb_entity_name",
        help="wandb entity name (usualy github ID)",
    )
    wandb_group.add_argument(
        "--wandb_project_name",
        help="wandb project name",
        default="blr_genetic_association",
    )
    wandb_group.add_argument(
        "--wandb_job_type",
        help="Wandb job type. This is useful for grouping runs together.",
        default=None,
    )
    args = parser.parse_args()

    ######      DDP initialization                #######
    if args.ddp:
        # Initialize distributed training
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
            # torchrun environment - use automatic initialization
            args.rank = int(os.environ['RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.local_rank = int(os.environ['LOCAL_RANK'])
            
            # Set device
            torch.cuda.set_device(args.local_rank)
            device = torch.device(f'cuda:{args.local_rank}')
            
            # Initialize process group with automatic rendezvous from torchrun
            dist.init_process_group(backend='nccl')
            
        elif 'SLURM_PROCID' in os.environ:
            # For SLURM environments (srun without torchrun)
            args.rank = int(os.environ['SLURM_PROCID'])
            args.world_size = int(os.environ['SLURM_NTASKS'])
            
            # Calculate local rank based on tasks per node
            if 'SLURM_LOCALID' in os.environ:
                args.local_rank = int(os.environ['SLURM_LOCALID'])
            else:
                # Fallback: assume equal distribution of tasks across nodes
                tasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', '1'))
                args.local_rank = args.rank % tasks_per_node
            
            # Get master node address
            if 'SLURM_LAUNCH_NODE_IPADDR' in os.environ:
                master_addr = os.environ['SLURM_LAUNCH_NODE_IPADDR']
            elif 'SLURM_NODELIST' in os.environ:
                # Parse first node from nodelist
                import subprocess
                result = subprocess.run(['scontrol', 'show', 'hostnames', os.environ['SLURM_NODELIST']], 
                                      capture_output=True, text=True)
                master_addr = result.stdout.strip().split('\n')[0]
            else:
                master_addr = args.master_addr
            
            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(args.local_rank)
                device = torch.device(f'cuda:{args.local_rank}')
            else:
                device = torch.device('cpu')
            
            # Initialize process group with manual TCP init for SLURM
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method=f'tcp://{master_addr}:{args.master_port}',
                world_size=args.world_size,
                rank=args.rank
            )
        else:
            # Fallback for manual setup
            args.rank = 0
            args.world_size = 1
            args.local_rank = 0
            
            # Set device
            if torch.cuda.is_available():
                torch.cuda.set_device(args.local_rank)
                device = torch.device(f'cuda:{args.local_rank}')
            else:
                device = torch.device('cpu')
            
            # Initialize process group with manual TCP init
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method=f'tcp://{args.master_addr}:{args.master_port}',
                world_size=args.world_size,
                rank=args.rank
            )
        
        # Ensure all processes sync
        dist.barrier()
        
        # Optional: Set rank-specific seeds for better randomness
        # Uncomment if you want each rank to have different random streams
        # rank_seed = base_seed + args.rank
        # torch.manual_seed(rank_seed)
        # np.random.seed(rank_seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(rank_seed)
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0

    ######      Logging setup                    #######
    if args.rank == 0:  # Only main process logs
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler(args.out + ".log", "w", "utf-8"),
                logging.StreamHandler()
            ]
        )

        logging.info(get_copyright_string())
        logging.info("")
        logging.info("Logs saved in: " + str(args.out + ".log"))
        logging.info("")

        logging.info("Options in effect: ")
        for arg in vars(args):
            logging.info('     {}: {}'.format(arg, getattr(args, arg)))

        logging.info("")
    else:
        # Disable logging for non-main processes
        logging.basicConfig(level=logging.CRITICAL)

    ######      Preprocessing the phenotypes      ######
    st = time.time()
    if args.rank == 0:
        logging.info("#### Start Time: " + str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')) + " ####")
        logging.info("")

    warnings.simplefilter("ignore")

    make_sure_path_exists(args.out)

    if args.out_step0 is not None:
        if args.rank == 0:
            logging.info("#### Step 1a. Using preprocessed phenotype and hdf5 files ####")
        rhe_out = args.out_step0
        hdf5_filename = args.out_step0 + ".hdf5"
        sample_indices = np.array(h5py.File(hdf5_filename, "r")["sample_indices"])
        if args.rank == 0:
            logging.info("#### Step 1a. Done in " + str(time.time() - st) + " secs ####")
            logging.info("")
    else:
        if args.rank == 0:
            logging.info("#### Step 1a. Preprocessing the phenotypes and converting bed to hdf5 ####")
        rhe_out = args.out
        Traits, covar_effects, sample_indices = preprocess_phenotypes(
            args.phenoFile, args.covarFile, args.bed, args.keepFile, args.binary, args.hdf5, args.phen_thres
        )
        PreparePhenoRHE(Traits, covar_effects, args.bed, rhe_out, None)
        hdf5_filename = convert_to_hdf5(
            args.bed,
            args.covarFile,
            sample_indices,
            args.out,
            args.modelSnps,
            args.hdf5,
            args.chunksize
        )
        if args.rank == 0:
            logging.info("#### Step 1a. Done in " + str(time.time() - st) + " secs ####")
            logging.info("")

    ######      Run RHE-MC for h2 estimation      ######
    if args.h2_file is None and not args.h2_grid:
        st = time.time()
        if args.rank == 0:
            logging.info("#### Step 1b. Calculating heritability estimates using RHE ####")
        args.annot = args.out + ".maf2_ld4.annot"
        MakeAnnotation(
            args.bed,
            args.ldscores,
            args.modelSnps,
            [0.01, 0.05, 0.5],
            [0.0, 0.25, 0.5, 0.75, 1.0],
            args.annot,
        )
        VC = runRHE(
            args.bed,
            rhe_out + ".rhe",
            args.modelSnps,
            args.annot,
            args.out + ".rhe.log",
            args.rhemc,
            args.covarFile,
            args.out,
            args.binary,
            args.rhe_random_vectors,
            args.rhe_jn
        )
        if args.rank == 0:
            logging.info("#### Step 1b. Done in " + str(time.time() - st) + " secs ####")
            logging.info("")
    elif args.h2_file is not None:
        st = time.time()
        if args.rank == 0:
            logging.info("#### Step 1b. Loading heritability estimates from: " + str(args.h2_file) + " ####")
            logging.info("")
        VC = np.loadtxt(args.h2_file)
        if args.rank == 0:
            logging.info("#### Step 1b. Done in " + str(time.time() - st) + " secs ####")
            logging.info("")
    else:
        st = time.time()
        if args.rank == 0:
            logging.info("#### Step 1b. Using h2_grid and performing a grid search in BLR ####")
            logging.info("")
        VC = None
        if args.rank == 0:
            logging.info("#### Step 1b. Done in " + str(time.time() - st) + " secs ####")
            logging.info("")


    ######      Running variational inference     ######
    st = time.time()
    if args.rank == 0:  # Only main process logs
        logging.info("#### Step 1c. Running VI using spike and slab prior ####")
    
    if args.ddp:
        # For DDP, device was already set during initialization
        device = torch.device(f'cuda:{args.local_rank}')
        if args.rank == 0:
            logging.info(f"Using DDP with {args.world_size} GPUs for variational inference!!")
            logging.info("")
    elif torch.cuda.is_available():
        if args.rank == 0:
            logging.info("Using GPU to run variational inference!!")
            logging.info("")
        device = 'cuda' 
    elif torch.backends.mps.is_available():
        if args.rank == 0:
            logging.info("Using MPS to run variational inference!!")
            logging.info("")
        device = 'mps'
    else:
        if args.rank == 0:
            logging.info("PyTorch can not detect either CUDA or MPS. Falling back to CPU to run variational"
                         " inference... expect very slow multiplications.")
            logging.info("")
        device = 'cpu'
            
    if args.kinship is None and args.rank == 0:
        logging.info("Caution: A kinship file wasn't supplied, no correction for relatives will be performed... this could lead to inflation if there are relatives in the dataset")
        logging.info("")
    
    beta = blr_spike_slab(args, VC, hdf5_filename, device)
    
    if args.ddp:
        # Clean up DDP
        dist.destroy_process_group()
    
    if args.rank == 0:  # Only main process logs and saves outputs
        logging.info("#### Step 1c. Done in " + str(time.time() - st) + " secs ####")
        logging.info("")

        logging.info("Saved LOCO predictions per phenotype as: ")
        with h5py.File(hdf5_filename,'r') as f:
            pheno_names = f['pheno_names'][:]
        with open(args.out + "_pred.list" , 'w') as f:
            for i, pheno_name in enumerate(pheno_names):
                f.write(pheno_name.decode() + " " + str(Path(args.out).resolve()) + "_" + str(i+1) + ".loco \n")
                logging.info(pheno_name.decode() + " : " + str(Path(args.out).resolve()) + "_" + str(i+1) + ".loco")
        logging.info("")
        logging.info("LOCO prediction locations per phenotype saved as: " + str(args.out + '_pred.list'))
        logging.info("")

        logging.info("Saved h2 estimates per phenotype as: " + str(args.out + '.h2'))
        logging.info("")
        logging.info("Saved sparsity estimates per phenotype as: " + str(args.out + '.alpha'))
        logging.info("")
        
    if args.predBetasFlag and args.rank == 0:
        snp_on_disk = Bed(args.bed, count_A1=True)
        if args.modelSnps is None:
            total_snps = snp_on_disk.sid_count
            snp_mask = np.ones(total_snps, dtype="bool")
        else:
            snps_to_keep = pd.read_csv(args.modelSnps, sep=r'\s+')
            snps_to_keep = snps_to_keep[snps_to_keep.columns[0]].values
            snp_dict = {}
            total_snps = snp_on_disk.sid_count
            snp_mask = np.zeros(total_snps, dtype="bool")
            for snp_no, snp in enumerate(snp_on_disk.sid):
                snp_dict[snp] = snp_no
            for snp in snps_to_keep:
                snp_mask[snp_dict[snp]] = True

        snp_on_disk = snp_on_disk[:, snp_mask]
        df = pd.DataFrame(columns = ['CHR','GENPOS','POS', 'SNP','BETA'])
        df['CHR'] = snp_on_disk.pos[:, 0]
        df['GENPOS'] = snp_on_disk.pos[:, 1]
        df['POS'] = snp_on_disk.pos[:, 2]
        df['SNP'] = snp_on_disk.sid
        bim = pd.read_csv(
            args.bed + ".bim",
            sep=r'\s+',
            header=None,
            names=["CHR", "SNP", "GENPOS", "POS", "A1", "A2"],
            dtype={"SNP":str}
        )
        bim = bim[['CHR','SNP','A1','A2']]
        df = pd.merge(df, bim, on=['CHR','SNP'])
        if args.rank == 0:
            print(df.shape)
        with h5py.File(hdf5_filename,'r') as f:
            pheno_names = f['pheno_names'][:]
        for d, pheno_name in enumerate(pheno_names):
            df['BETA'] = beta[d]
            df.to_csv(args.out + '_' + pheno_name.decode() + '.posterior_betas', sep='\t', index=None, na_rep='NA')
        if args.rank == 0:
            logging.info("Saved prediction posterior betas per phenotype as: ")
            for i, pheno_name in enumerate(pheno_names):
                logging.info(pheno_name.decode() + " : " + str(Path(args.out).resolve()) + "_" + pheno_name.decode() + ".posterior_betas")
            logging.info("")

    if args.rank == 0:
        logging.info("#### Step 1 total Time: " + str(time.time() - overall_st) + " secs ####")
        logging.info("")
        logging.info("#### End Time: " + str(datetime.today().strftime('%Y-%m-%d %H:%M:%S')) + " ####")
        logging.info("")


if __name__ == "__main__":
    main()
