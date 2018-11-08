all: transE_train.cpp transE_test.cpp parTransE_train.cpp parTransE_test.cpp
	g++ -std=c++11 transE_train.cpp -o transE_train.bin -O3 -march=native
	g++ -std=c++11 transE_test.cpp -o transE_test.bin -O3 -march=native
	g++ -std=c++11 parTransE_train.cpp -o parTransE_train.bin -O3 -march=native --openmp
	g++ -std=c++11 parTransE_test.cpp -o parTransE_test.bin -O3 -march=native --openmp

seq:
	./transE_train.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./ --epochs 100

run:
	g++ -std=c++11 transE_train.cpp -o transE_train.bin -O3 -march=native
	./transE_train.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./ --epochs 10 --v

test:
	g++ -std=c++11 transE_test.cpp -o transE_test.bin -O3 -march=native
	./transE_test.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./ --v

par:
	g++ -std=c++11 parTransE_train.cpp -o parTransE_train.bin -O3 -march=native --openmp

parrun:
	g++ -std=c++11 parTransE_train.cpp -o parTransE_train.bin -O3 -march=native --openmp
	./parTransE_train.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ../ --epochs 10 --threads 8 --v

partest:
	g++ -std=c++11 parTransE_test.cpp -o parTransE_test.bin -O3 -march=native --openmp
	./parTransE_test.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./ --threads 72 --v

mpi:
	mpicxx -std=c++11 mpiTransE_train.cpp -o mpiTransE_train.bin -O3 -march=native --openmp

mpirun:
	mpicxx -std=c++11 mpiTransE_train.cpp -o mpiTransE_train.bin -O3 -march=native --openmp
	srun -N 2 ./mpiTransE_train.bin --size 50 --input ../Fast-TransX/data/FB15K/ --output ../ --epochs 10 --v

mpitest:
	mpicxx -std=c++11 mpiTransE_test.cpp -o mpiTransE_test.bin -O3 -march=native --openmp
	srun -N 2 ./mpiTransE_test.bin --size 50 --input ../Fast-TransX/data/FB15K/ --output ./ --v

ga:
	g++ -std=c++11 graphAnalyze.cpp -o graphAnalyze.bin -O3

clean:
	-rm *.bin
