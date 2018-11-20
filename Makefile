all: transE_train.cpp transE_test.cpp parTransE_train.cpp parTransE_test.cpp
	-@mkdir bin
	g++ -std=c++11 transE_train.cpp -o bin/transE_train.bin -O3 -march=native
	g++ -std=c++11 transE_test.cpp -o bin/transE_test.bin -O3 -march=native
	g++ -std=c++11 parTransE_train.cpp -o bin/parTransE_train.bin -O3 -march=native --openmp
	g++ -std=c++11 parTransE_test.cpp -o bin/parTransE_test.bin -O3 -march=native --openmp

seq:
	./bin/transE_train.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./result --epochs 100

run:
	-@mkdir bin
	g++ -std=c++11 transE_train.cpp -o bin/transE_train.bin -O3 -march=native
	./bin/transE_train.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./result --epochs 100 --v

test:
	-@mkdir bin
	g++ -std=c++11 transE_test.cpp -o bin/transE_test.bin -O3 -march=native
	./bin/transE_test.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./result --v

par:
	-@mkdir bin
	g++ -std=c++11 parTransE_train.cpp -o bin/parTransE_train.bin -O3 -march=native --openmp

parrun:
	-@mkdir bin
	g++ -std=c++11 parTransE_train.cpp -o bin/parTransE_train.bin -O3 -march=native --openmp
	./bin/parTransE_train.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./result --epochs 100 --threads 24 --v

partest:
	-@mkdir bin
	g++ -std=c++11 parTransE_test.cpp -o bin/parTransE_test.bin -O3 -march=native --openmp
	./bin/parTransE_test.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./result --threads 72 --v

mpi:
	-@mkdir bin
	mpicxx -std=c++11 mpiTransE_train.cpp -o bin/mpiTransE_train.bin -O3 -march=native --openmp

mpirun:
	-@mkdir bin
	mpicxx -std=c++11 mpiTransE_train.cpp -o bin/mpiTransE_train.bin -O3 -march=native --openmp
	srun -N 2 ./bin/mpiTransE_train.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./result --epochs 50 --v

mpitest:
	-@mkdir bin
	mpicxx -std=c++11 mpiTransE_test.cpp -o bin/mpiTransE_test.bin -O3 -march=native --openmp
	srun -N 2 ./bin/mpiTransE_test.bin --size 50 --input ../Fast-TransX/data/FB15K/ --output ./result --v

ga:
	-@mkdir bin
	g++ -std=c++11 graphAnalyze.cpp -o bin/graphAnalyze.bin -O3

clean:
	-rm bin/*.bin
	-rmdir bin
