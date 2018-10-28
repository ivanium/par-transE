all: transE
	g++ transE_train.cpp -o transE.bin -O3 -march=native

run:
	./transE.bin --size 50 --input ../Fast-TransX/data/FB15K/ --output ./ --epochs 100

rerun:
	g++ transE_train.cpp -o transE.bin -O3 -march=native
	./transE.bin --size 50 --input ../Fast-TransX/data/FB15K/ --output ./ --epochs 10 --v

par:
	g++ parTransE_train.cpp -o transE.bin -O3 -march=native

clean:
	-rm *.bin
