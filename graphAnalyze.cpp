#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <string>
#include <algorithm>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <iostream>

#define ENTITY_MAX 1000000
#define RELATION_MAX 1000000

using namespace std;

string filePath;
int tripleNum, entityNum, relationNum;

int hEBuf[ENTITY_MAX], tEBuf[ENTITY_MAX], rBuf[RELATION_MAX];

void init() {
  filePath = "";
}

void load() {
  FILE *fin;
  fin = fopen(filePath.c_str(), "r");
  fscanf(fin, "%d", &tripleNum);
  int h, t, r;
  for (int i = 0; i < tripleNum; i++) {
    fscanf(fin, "%d %d %d", &h, &t, &r);
    hEBuf[h]++;
    tEBuf[t]++;
    rBuf[r]++;
    if (h > entityNum) {
      entityNum = h;
    }
    if (t > entityNum) {
      entityNum = t;
    }
    if (r > relationNum) {
      relationNum = r;
    }
  }
  fclose(fin);
}

void output() {
  FILE *fout;
  fout = fopen("entityAnalysis.txt", "w");
  fprintf(fout, "%d\n", entityNum);
  for (int i = 0; i < entityNum; i++) {
    fprintf(fout, "%d\t%d\t%d\n", hEBuf[i], tEBuf[i], hEBuf[i] + tEBuf[i]);
  }
  fclose(fout);

  fout = fopen("relationAnalysis.txt", "w");
  fprintf(fout, "%d\n", relationNum);
  for (int i = 0; i < relationNum; i++) {
    fprintf(fout, "%d\n", rBuf[i]);
  }
  fclose(fout);
}

int main(int argc, char *argv[]) {
  init();
  filePath = "../Fast-TransX/data/FB15K/train2id.txt";
  load();
  output();
  return 0;
}
