all: transE
	g++ transE.cpp -o transE.bin -O3 -march=native

run:
	./transE.bin -size 50 -input ../Fast-TransX/data/FB15K/ -output ./ -thread 1 -epochs 100

rerun:
	g++ transE.cpp -o transE.bin -O3 -march=native
	./transE.bin -size 50 -input ../Fast-TransX/data/FB15K/ -output ./ -thread 1 -epochs 10

clean:
	-rm *.bin
