echo "node,process,cal,com,IO"
for node in {1..4}; do
    for pro in {2..12}; do
        if [ "$((pro % node))" -eq 0 ]; then
            echo -n "$node,$pro,"
            srun -N$node -n$pro ./hw1 536869888 testcases/35.in 35.out
        fi
    done
done