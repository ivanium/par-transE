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

struct Node {
  int id, inD, outD;
};

struct cmpIn {
  bool operator() (const Node &a, const Node &b) {
    return (a.inD > b.inD ||
           (a.inD == b.inD && a.outD > b.outD) ||
           (a.inD == b.inD && a.outD == b.outD && a.id < b.id));
  }
};

struct cmpOut {
  bool operator() (const Node &a, const Node &b) {
    return (a.outD > b.outD ||
           (a.outD == b.outD && a.inD > b.inD) ||
           (a.outD == b.outD && a.inD == b.inD && a.id < b.id));
  }
};

struct Edge {
  int id, cnt;
};

struct cmpR {
  bool operator() (const Edge &a, const Edge &b) {
    return (a.cnt > b.cnt ||
           (a.cnt == b.cnt && a.id < b.id));
  }
};

struct Node nodeBuf[ENTITY_MAX];
struct Edge rBuf[RELATION_MAX];

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
    nodeBuf[h].outD++;
    nodeBuf[t].inD++;
    rBuf[r].cnt++;
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
  for (int i = 0; i < entityNum; i++) {
    nodeBuf[i].id = i;
  }
  sort(nodeBuf, nodeBuf + entityNum, cmpOut());

  fout = fopen("entityAnalysis.txt", "w");
  fprintf(fout, "%d\n", entityNum);
  for (int i = 0; i < entityNum; i++) {
    fprintf(fout, "%d\t%d\t%d\t%d\n", nodeBuf[i].id, nodeBuf[i].outD, nodeBuf[i].inD, nodeBuf[i].inD + nodeBuf[i].outD);
  }
  fclose(fout);

  fprintf(fout, "%d\n", relationNum);
  for (int i = 0; i < relationNum; i++) {
    rBuf[i].id = i;
  }
  sort(rBuf, rBuf + relationNum, cmpR());

  fout = fopen("relationAnalysis.txt", "w");
  for (int i = 0; i < relationNum; i++) {
    fprintf(fout, "%d\t%d\n", rBuf[i].id, rBuf[i].cnt);
  }
  fclose(fout);
}

int main(int argc, char *argv[]) {
  init();
  // filePath = "./triple2id.txt";
  filePath = "../Fast-TransX/data/FB15K/train2id.txt";
  load();
  output();
  return 0;
}
