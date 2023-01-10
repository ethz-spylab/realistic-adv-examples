echo "Running the following command:"
echo "python main.py --early 0 --out-dir /cluster/project/privsec/realistic-adv-examples --max-unsafe-queries 15000 --max-queries 100000 $1"

sbatch --ntasks 4 --mem-per-cpu 5G --ngpus 1 --gres=gpumem:24G -t 120:00:00 -o logs/slurm-%A_%a.out --wrap "python main.py --early 0 --out-dir /cluster/project/privsec/realistic-adv-examples --max-unsafe-queries 15000 --max-queries 100000 $1"