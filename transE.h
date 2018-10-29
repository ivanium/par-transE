#ifndef TRANSE_H
#define TRANSE_H

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

#include "parallel.h"
#include "util.h"

using namespace std;

struct Triple {
  intT h, r, t;
};

struct cmp_head {
  bool operator() (const Triple &a, const Triple &b) {
    return (a.h < b.h) ||
           (a.h == b.h && a.r < b.r) ||
           (a.h == b.h && a.r == b.r && a.t < b.t);
  }
};

struct cmp_tail {
  bool operator() (const Triple &a, const Triple &b) {
    return (a.t < b.t) ||
           (a.t == b.t && a.r < b.r) ||
           (a.t == b.t && a.r == b.r && a.h < b.h);
  }
};

// Hyperparameters
floatT alpha  = 0.001;
floatT margin = 1.0;
intT dimension= 100;
intT bernFlag = 0;
intT epochs   = 1000;
intT nbatches = 1;

// Arguments
intT loadBinaryFlag = 0;
intT outBinaryFlag = 0;
string inputDir = "./";
string outputDir = "";
string loadDir = "";
string note = "";

// Buffer
Triple *trainHead, *trainTail, *trainList;
intT *headBegins, *headEnds,
     *tailBegins, *tailEnds;

intT relationNum, entityNum, tripleNum;
floatT *vecBuf, *rVecBuf, *eVecBuf;
floatT *headMeanList, *tailMeanList;

// Global Variable
floatT res = 0.0;
intT batchSize;
ULL seed = 0x1f;


// TRAIN
void trainInit() {
  struct timeval stt; gettimeofday(&stt, NULL);
  printf("START TRAIN INITIALIZATION ...\n");

  FILE *fin;
  intT tmp;

  fin = fopen((inputDir + "relation2id.txt").c_str(), "r");
  tmp = fscanf(fin, "%d", &relationNum);
  fclose(fin);

  fin = fopen((inputDir + "entity2id.txt").c_str(), "r");
  tmp = fscanf(fin, "%d", &entityNum);
  fclose(fin);

  // setup buffer variables
  vecBuf = (floatT *) malloc((entityNum + relationNum) * dimension * sizeof(floatT));
  rVecBuf = vecBuf;
  eVecBuf = vecBuf + relationNum * dimension;

  intT *rFreqList = (intT *) malloc(relationNum * sizeof(intT));

  headBegins = (intT *) malloc(entityNum*4 * sizeof(intT));
  headEnds   = headBegins + entityNum;
  tailBegins = headEnds   + entityNum;
  tailEnds   = tailBegins + entityNum;

  headMeanList = (floatT *) malloc(relationNum * 2 * sizeof(floatT));
  tailMeanList = headMeanList + relationNum;

  fin = fopen((inputDir + "train2id.txt").c_str(), "r");
  tmp = fscanf(fin, "%d", &tripleNum);
  trainList = (Triple *) malloc(tripleNum*3 * sizeof(Triple));
  trainHead = trainList + tripleNum;
  trainTail = trainHead + tripleNum;

  // Initialize vectors and calc stats
  for (intT i = 0; i < relationNum; i++) {
    for (int ii = 0; ii < dimension; ii++) {
      rVecBuf[i*dimension + ii] = randn(&seed, 0.0, 1.0/dimension, -6/sqrt(dimension), 6/sqrt(dimension));
    }
  }
  for (intT i = 0; i < entityNum; i++) {
    for (int ii = 0; ii < dimension; ii++) {
      eVecBuf[i*dimension + ii] = randn(&seed, 0.0, 1.0/dimension, -6/sqrt(dimension), 6/sqrt(dimension));
    }
    norm(eVecBuf + i*dimension, dimension);
  }

  for (intT i = 0; i < tripleNum; i++) {
    tmp = fscanf(fin, "%d %d %d", &trainList[i].h, &trainList[i].t, &trainList[i].r);
    trainHead[i] = trainList[i];
    trainTail[i] = trainList[i];
  }
  fclose(fin);
  // memcpy(trainHead, trainList, tripleNum * sizeof(Triple));
  // memcpy(trainTail, trainList, tripleNum * sizeof(Triple));

  sort(trainHead, trainHead + tripleNum, cmp_head());
  sort(trainTail, trainTail + tripleNum, cmp_tail());

  for (intT i = 0; i < tripleNum; i++) {
    rFreqList[trainHead[i].r]++;
  }

  headBegins[0] = tailBegins[0] = 0;
  memset(headEnds, -1, sizeof(intT) * entityNum);
  memset(tailEnds, -1, sizeof(intT) * entityNum);

  for (intT i = 1; i < tripleNum; i++) {
    if (trainHead[i-1].h != trainHead[i].h) {
      headEnds[trainHead[i-1].h] = i-1;
      headBegins[trainHead[i].h] = i;
    }
  }
  for (intT i = 1; i < tripleNum; i++) {
    if (trainTail[i-1].t != trainTail[i].t) {
      tailEnds[trainTail[i-1].t] = i-1;
      tailBegins[trainTail[i].t] = i;
    }
  }
  headEnds[trainHead[tripleNum -1].h] = tripleNum - 1;
  tailEnds[trainTail[tripleNum -1].t] = tripleNum - 1;

  for (intT i = 0; i < entityNum; i++) {
    for (intT j = headBegins[i] + 1; j <= headEnds[i]; j++)
      if (trainHead[j].r != trainHead[j - 1].r)
        headMeanList[trainHead[j].r] += 1.0;
    if (headBegins[i] <= headEnds[i])
      headMeanList[trainHead[headBegins[i]].r] += 1.0;
    for (intT j = tailBegins[i] + 1; j <= tailEnds[i]; j++)
      if (trainTail[j].r != trainTail[j - 1].r)
        tailMeanList[trainTail[j].r] += 1.0;
    if (tailBegins[i] <= tailEnds[i])
      tailMeanList[trainTail[tailBegins[i]].r] += 1.0;
  }

  for (intT i = 0; i < relationNum; i++) {
    headMeanList[i] = rFreqList[i] / headMeanList[i];
    tailMeanList[i] = rFreqList[i] / tailMeanList[i];
  }
  // cleanup temp buffer
  free (rFreqList);
  struct timeval end; gettimeofday(&end, NULL);
  printf("FINISH TRAIN INITIALIZATION, INIT duration: %.3fs\n", (end.tv_sec-stt.tv_sec) + (end.tv_usec-stt.tv_usec)/1e6);
}

void trainFinish() {
  free (vecBuf);
  free (headBegins);
  // free (headEnds);
  // free (tailBegins); free (tailEnds);
  free (headMeanList);
  free (trainList);
  // free (trainHead); free (trainTail);
}

inline floatT tripleDiff(floatT *hVec, floatT *tVec, floatT *rVec) {
  floatT sum = 0.0;
  for (int ii = 0; ii < dimension; ii++) {
    sum += fabs(tVec[ii] - hVec[ii] - rVec[ii]);
  }
  return sum;
}

inline void gradiant(floatT *h1Vec, floatT *t1Vec, floatT *rVec, floatT *jVec, bool jHeadFlag) {
  floatT *h2Vec, *t2Vec;
  if (jHeadFlag) {
    h2Vec = jVec;
    t2Vec = t1Vec;
  } else {
    h2Vec = h1Vec;
    t2Vec = jVec;
  }
  for (int ii = 0; ii < dimension; ii++) {
    floatT x;
    x = (t1Vec[ii] - h1Vec[ii] - rVec[ii]);
    x = x > 0 ? -alpha : alpha;
    rVec[ii]  -= x;
    h1Vec[ii] -= x;
    t1Vec[ii] += x;
    x = (t2Vec[ii] - h2Vec[ii] - rVec[ii]);
    x = x > 0 ? alpha : -alpha;
    rVec[ii]  -= x;
    h2Vec[ii] -= x;
    t2Vec[ii] += x;
  }
}

inline void tripleTrain(floatT *h1Vec, floatT *t1Vec, floatT *rVec, floatT *jVec, bool jHeadFlag) {
  floatT *h2Vec, *t2Vec;
  if (jHeadFlag) {
    h2Vec = jVec;
    t2Vec = t1Vec;
  } else {
    h2Vec = h1Vec;
    t2Vec = jVec;
  }
  floatT sum1 = tripleDiff(h1Vec, t1Vec, rVec);
  floatT sum2 = tripleDiff(h2Vec, t2Vec, rVec);
  if (sum1 + margin > sum2) {
    res += margin + sum1 - sum2;
    gradiant(h1Vec, t1Vec, rVec, jVec, jHeadFlag);
  }
}

inline intT getNegTail(ULL *id, intT h, intT r) {
  intT lef, rig, mid, ll, rr;
  lef = headBegins[h]-1;
  rig = headEnds[h];
  while (lef + 1 < rig) {
    mid = (lef + rig) >> 1;
    if (trainHead[mid].r >= r)
      rig = mid;
    else
      lef = mid;
  }
  ll = rig;
  lef = headBegins[h];
  rig = headEnds[h]+1;
  while (lef + 1 < rig) {
    mid = (lef + rig) >> 1;
    if (trainHead[mid].r <= r)
      lef = mid;
    else
      rig = mid;
  }
  rr = lef;
  intT tmp = rand_max(id, entityNum - (rr - ll + 1));
  if (tmp < trainHead[ll].t) return tmp;
  if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
  lef = ll, rig = rr + 1;
  while (lef + 1 < rig) {
    mid = (lef + rig) >> 1;
    if (trainHead[mid].t - mid + ll - 1 < tmp)
      lef = mid;
    else 
      rig = mid;
  }
  return tmp + lef - ll + 1;
}

inline intT getNegHead(ULL *id, intT t, intT r) {
  intT lef, rig, mid, ll, rr;
  lef = tailBegins[t] - 1;
  rig = tailEnds[t];
  while (lef + 1 < rig) {
    mid = (lef + rig) >> 1;
    if (trainTail[mid].r >= r)
      rig = mid;
    else
      lef = mid;
  }
  ll = rig;
  lef = tailBegins[t];
  rig = tailEnds[t] + 1;
  while (lef + 1 < rig) {
    mid = (lef + rig) >> 1;
    if (trainTail[mid].r <= r)
      lef = mid;
    else
      rig = mid;
  }
  rr = lef;
  intT tmp = rand_max(id, entityNum - (rr - ll + 1));
  if (tmp < trainTail[ll].h) return tmp;
  if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
  lef = ll, rig = rr + 1;
  while (lef + 1 < rig) {
    mid = (lef + rig) >> 1;
    if (trainTail[mid].h - mid + ll - 1 < tmp)
      lef = mid;
    else 
      rig = mid;
  }
  return tmp + lef - ll + 1;
}

void train_thread() {
  floatT *hVec, *tVec, *rVec, *jVec;
  for (intT pr, i, j, k = batchSize; k >= 0; k--) {
    i = rand_max(&seed, tripleNum);

    pr = bernFlag ? 1000 * headMeanList[trainList[i].r] / (headMeanList[trainList[i].r] + tailMeanList[trainList[i].r])
                  : 500;

    bool jHeadFlag = !(rand_max(&seed, 1000) < pr);
    j = jHeadFlag ? getNegHead(&seed, trainList[i].t, trainList[i].r)
                  : getNegTail(&seed, trainList[i].h, trainList[i].r);

    hVec = eVecBuf + dimension * trainList[i].h;
    tVec = eVecBuf + dimension * trainList[i].t;
    rVec = rVecBuf + dimension * trainList[i].r;
    jVec = eVecBuf + dimension * j;

    tripleTrain(hVec, tVec, rVec, jVec, jHeadFlag);
    norm(hVec, dimension); norm(tVec, dimension); norm(rVec, dimension); norm(jVec, dimension);
  }
}

void train() {
  struct timeval stt; gettimeofday(&stt, NULL);
  printf("START TRAINING ...\n");

  batchSize = tripleNum / nbatches;
  floatT *hVec, *tVec, *rVec, *jVec;
  for (intT e = 0; e < epochs; e++) {
    res = 0.0;
    for (intT batch = 0; batch < nbatches; batch++) {
      for (intT pr, i, j, k = 0; k <= batchSize; k++) {
        i = rand_max(&seed, tripleNum);
        // i = k;

        pr = bernFlag ? 1000 * headMeanList[trainList[i].r] / (headMeanList[trainList[i].r] + tailMeanList[trainList[i].r])
                      : 500;

        bool jHeadFlag = rand_max(&seed, 1000) >= pr;
        j = jHeadFlag ? getNegHead(&seed, trainList[i].t, trainList[i].r)
                      : getNegTail(&seed, trainList[i].h, trainList[i].r);

        hVec = eVecBuf + dimension * trainList[i].h;
        tVec = eVecBuf + dimension * trainList[i].t;
        rVec = rVecBuf + dimension * trainList[i].r;
        jVec = eVecBuf + dimension * j;

        tripleTrain(hVec, tVec, rVec, jVec, jHeadFlag);
        norm(hVec, dimension); norm(tVec, dimension); norm(rVec, dimension); norm(jVec, dimension);
      }
    }
    printf("epoch %d %f\n", e, res);
  }

  struct timeval end; gettimeofday(&end, NULL);
  printf("END TRAINING. TRAIN duration: %.3fs\n", (end.tv_sec-stt.tv_sec) + (end.tv_usec-stt.tv_usec)/(1e6));
}


// Load vec file and output train result
void load() {
  if (loadBinaryFlag) {
    struct stat statbuf;
    if (stat((loadDir + "entity2vec" + note + ".bin").c_str(), &statbuf) != -1) {  
      intT fd = open((loadDir + "entity2vec" + note + ".bin").c_str(), O_RDONLY);
      floatT* eVecTmp = (floatT*)mmap(NULL, statbuf.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
      memcpy(eVecBuf, eVecTmp, statbuf.st_size);
      munmap(eVecTmp, statbuf.st_size);
      close(fd);
    }
    if (stat((loadDir + "relation2vec" + note + ".bin").c_str(), &statbuf) != -1) {  
      intT fd = open((loadDir + "relation2vec" + note + ".bin").c_str(), O_RDONLY);
      floatT* rVecTmp =(floatT*)mmap(NULL, statbuf.st_size, PROT_READ, MAP_PRIVATE, fd, 0); 
      memcpy(rVecBuf, rVecTmp, statbuf.st_size);
      munmap(rVecTmp, statbuf.st_size);
      close(fd);
    }
  } else {
    FILE *fin;
    int tmp;
    fin = fopen((loadDir + "entity2vec" + note + ".vec").c_str(), "r");

    int offset = 0;
    for (intT i = 0; i < entityNum; i++) {
      for (intT ii = 0; ii < dimension; ii++) {
        tmp = fscanf(fin, "%f", &eVecBuf[offset + ii]);
      }
      offset += dimension;
    }
    fclose(fin);
    offset = 0;
    fin = fopen((loadDir + "relation2vec" + note + ".vec").c_str(), "r");
    for (intT i = 0; i < relationNum; i++) {
      for (intT ii = 0; ii < dimension; ii++) {
        tmp = fscanf(fin, "%f", &rVecBuf[offset + ii]);
      }
      offset += dimension;
    }
    fclose(fin);
  }
}

void output() {
  if (outBinaryFlag) {
    intT len, tot;
    floatT *head;
    FILE* f2 = fopen((outputDir + "relation2vec" + note + ".bin").c_str(), "wb");
    FILE* f3 = fopen((outputDir + "entity2vec" + note + ".bin").c_str(), "wb");
    len = relationNum * dimension; tot = 0;
    head = rVecBuf;
    while (tot < len) {
      intT sum = fwrite(head + tot, sizeof(floatT), len - tot, f2);
      tot += sum;
    }
    len = entityNum * dimension; tot = 0;
    head = eVecBuf;
    while (tot < len) {
      intT sum = fwrite(head + tot, sizeof(floatT), len - tot, f3);
      tot += sum;
    }	
    fclose(f2);
    fclose(f3);
  } else {
    FILE* f2 = fopen((outputDir + "relation2vec" + note + ".vec").c_str(), "w");
    FILE* f3 = fopen((outputDir + "entity2vec" + note + ".vec").c_str(), "w");
    intT offset = 0;
    for (intT i = 0; i < relationNum; i++) {
      for (intT ii = 0; ii < dimension; ii++) {
        fprintf(f2, "%.6f\t", rVecBuf[offset + ii]);
      }
      fprintf(f2,"\n");
      offset += dimension;
    }
    fclose(f2);
    offset = 0;
    for (intT i = 0; i < entityNum; i++) {
      for (intT ii = 0; ii < dimension; ii++) {
        fprintf(f3, "%.6f\t", eVecBuf[offset + ii]);
      }
      fprintf(f3,"\n");
      offset += dimension;
    }
    fclose(f3);
  }
}

#endif // !TRANSE_H