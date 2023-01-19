for dir in $(cat "exp_directories.txt"); do
    echo "sbatch -n 4 --mem-per-cpu=8G -o logs/$dir-conversion.log --wrap \"python plot_dist_vs_queries.py --exp-paths $dir --out-path /tmp/test.png --checksum-check\"" 
    sbatch -n 4 --mem-per-cpu=8G -o logs/$dir-conversion.log --wrap "python plot_dist_vs_queries.py --exp-paths $dir --out-path /tmp/test.png --checksum-check"
    # echo "sbatch -n 4 --mem-per-cpu=8G -o logs/$dir-unsafe-conversion.log --wrap \"python plot_dist_vs_queries.py --exp-paths $dir --out-path /tmp/test.png --checksum-check --unsafe-only\"" 
    # sbatch -n 4 --mem-per-cpu=8G -o logs/$dir-unsafe-conversion.log --wrap "python plot_dist_vs_queries.py --exp-paths $dir --out-path /tmp/test.png --checksum-check --unsafe-only" 
done