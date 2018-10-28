all: transE
	g++ -std=c++11 transE_train.cpp -o transE.bin -O3 -march=native

run:
	./transE.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./ --epochs 100

rerun:
	g++ -std=c++11 transE_train.cpp -o transE.bin -O3 -march=native
	./transE.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./ --epochs 10 --v

par:
	g++ -std=c++11 parTransE_train.cpp -o transE.bin -O3 -march=native --openmp

parrun:
	g++ -std=c++11 parTransE_train.cpp -o transE.bin -O3 -march=native --openmp
	./transE.bin --size 128 --input ../Fast-TransX/data/FB15K/ --output ./ --epochs 1000 --threads 1 --v

clean:
	-rm *.bin
