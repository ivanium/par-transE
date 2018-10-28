#ifndef PARTRANSE_H
#define PARTRANSE_H

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
#include <omp.h>

#include "parallel.h"
#include "util.h"

#include <iostream>

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
string inputBaseDir = "./";
string outputBaseDir = "";
string loadDir = "";

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

void init() {
  clock_t stt = clock(); printf("START INITIALING...\n");

  FILE *fin;
  intT tmp;

  fin = fopen((inputBaseDir + "relation2id.txt").c_str(), "r");
  tmp = fscanf(fin, "%d", &relationNum);
  fclose(fin);

  fin = fopen((inputBaseDir + "entity2id.txt").c_str(), "r");
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

  fin = fopen((inputBaseDir + "train2id.txt").c_str(), "r");
  tmp = fscanf(fin, "%d", &tripleNum);
  trainList = (Triple *) malloc(tripleNum*3 * sizeof(Triple));
  trainHead = trainList + tripleNum;
  trainTail = trainHead + tripleNum;

  // Initialize vectors and calc stats
  #pragma omp parallel for
  for (intT i = 0; i < relationNum; i++) {
    ULL seed = omp_get_thread_num();
    for (int ii = 0; ii < dimension; ii++) {
      rVecBuf[i*dimension + ii] = randn(&seed, 0.0, 1.0/dimension, -6/sqrt(dimension), 6/sqrt(dimension));
    }
  }
  #pragma omp parallel for
  for (intT i = 0; i < entityNum; i++) {
    ULL seed = omp_get_thread_num();
    for (int ii = 0; ii < dimension; ii++) {
      eVecBuf[i*dimension + ii] = randn(&seed, 0.0, 1.0/dimension, -6/sqrt(dimension), 6/sqrt(dimension));
    }
    norm(eVecBuf + i*dimension, dimension);
  }

  for (intT i = 0; i < tripleNum; i++) {
    tmp = fscanf(fin, "%d %d %d", &trainList[i].h, &trainList[i].t, &trainList[i].r);
  }
  fclose(fin);
  memcpy(trainHead, trainList, tripleNum * sizeof(Triple));
  memcpy(trainTail, trainList, tripleNum * sizeof(Triple));

  sort(trainHead, trainHead + tripleNum, cmp_head());
  sort(trainTail, trainTail + tripleNum, cmp_tail());

  #pragma omp parallel for
  for (intT i = 0; i < tripleNum; i++) {
    #pragma omp atomic
    rFreqList[trainHead[i].r]++;
  }

  headBegins[0] = tailBegins[0] = 0;
  memset(headEnds, -1, sizeof(intT) * entityNum);
  memset(tailEnds, -1, sizeof(intT) * entityNum);

  #pragma omp parallel for
  for (intT i = 1; i < tripleNum; i++) {
    if (trainHead[i-1].h != trainHead[i].h) {
      headEnds[trainHead[i-1].h] = i-1;
      headBegins[trainHead[i].h] = i;
    }
  // }
  // for (intT i = 1; i < tripleNum; i++) {
    if (trainTail[i-1].t != trainTail[i].t) {
      tailEnds[trainTail[i-1].t] = i-1;
      tailBegins[trainTail[i].t] = i;
    }
  }
  headEnds[trainHead[tripleNum -1].h] = tripleNum - 1;
  tailEnds[trainTail[tripleNum -1].t] = tripleNum - 1;

  #pragma omp parallel for
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

  #pragma omp parallel for
  for (intT i = 0; i < relationNum; i++) {
    headMeanList[i] = rFreqList[i] / headMeanList[i];
    tailMeanList[i] = rFreqList[i] / tailMeanList[i];
  }
  // cleanup temp buffer
  free (rFreqList);
  clock_t end = clock(); printf("FINISH INIT, duration: %.3fs\n", 1.0 * (end-stt) / CLOCKS_PER_SEC);
}

void finish() {
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

void* train_thread(void *con) {
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
  return NULL;
}

void train() {
  clock_t stt = clock(); printf("START TRAINING...\n");

  batchSize = tripleNum / nbatches;
  for (intT e = 0; e < epochs; e++) {
    res = 0.0;
    for (intT batch = 0; batch < nbatches; batch++) {
      train_thread((void *) &e);
    }
    printf("epoch %d %f\n", e, res);
  }

  clock_t end = clock(); printf("END TRAINING. Duration: %.3fs\n", 1.0 * (end - stt) / CLOCKS_PER_SEC);
}

#endif // !PARTRANSE_H