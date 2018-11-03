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
#include <sys/time.h>
#include <omp.h>

#include "parallel.h"
#include "util.h"

using namespace std;

struct Triple {
  intT h, r, t;
};

struct LabelTriple {
  intT label;
  intT h, r, t;
};

struct cmp_head {
  bool operator() (const Triple &a, const Triple &b) {
    return (a.h < b.h) ||
           (a.h == b.h && a.r < b.r) ||
           (a.h == b.h && a.r == b.r && a.t < b.t);
  }
  bool operator() (const LabelTriple &a, const LabelTriple &b) {
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
  bool operator() (const LabelTriple &a, const LabelTriple &b) {
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
int threads = 32;

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
intT blockSize;
ULL *seed;


void trainInit() {
  struct timeval stt; gettimeofday(&stt, NULL);
  printf("START INITIALING...\n");

  omp_set_num_threads(threads);
  seed = (ULL *) malloc(threads * sizeof(ULL));
  for (int i = 1; i <= threads; i++) {
    seed[i] = i;
  }

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
  #pragma omp parallel for
  for (intT i = 0; i < relationNum; i++) {
    ULL seed = i;
    for (int ii = 0; ii < dimension; ii++) {
      rVecBuf[i*dimension + ii] = randn(&seed, 0.0, 1.0/dimension, -6/sqrt(dimension), 6/sqrt(dimension));
    }
  }
  #pragma omp parallel for
  for (intT i = 0; i < entityNum; i++) {
    ULL seed = i;
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
    // rFreqList[trainHead[i].r]++;
    __sync_fetch_and_add(rFreqList + trainHead[i].r, 1);
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
  struct timeval end; gettimeofday(&end, NULL);
  printf("FINISH INIT, duration: %.3fs\n", (end.tv_sec-stt.tv_sec) + (end.tv_usec-stt.tv_usec)/1e6);
}

void trainFinish() {
  free (vecBuf);
  free (headBegins);
  free (headMeanList);
  free (trainList);
  free (seed);
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
    #pragma omp atomic
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

void train_thread(int tid) {
  intT begin = tid*blockSize;
  intT end   = tid == threads-1 ? batchSize : begin + blockSize;
  for (intT k = begin; k < end; k++) {
    intT pr, i, j;
    floatT *hVec, *tVec, *rVec, *jVec;
    // i = rand_max(&seed[tid], tripleNum);
    i = k;

    pr = bernFlag ? 1000 * headMeanList[trainList[i].r] / (headMeanList[trainList[i].r] + tailMeanList[trainList[i].r])
                  : 500;

    bool jHeadFlag = !(rand_max(&seed[tid], 1000) < pr);
    j = jHeadFlag ? getNegHead(&seed[tid], trainList[i].t, trainList[i].r)
                  : getNegTail(&seed[tid], trainList[i].h, trainList[i].r);

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
  printf("START TRAINING...\n");

  batchSize = tripleNum / nbatches;
  blockSize = batchSize / threads;
  for (intT e = 0; e < epochs; e++) {
    res = 0.0;
    for (intT batch = 0; batch < nbatches; batch++) {
      #pragma omp parallel for reduction(+:res)
      for (int i = 0; i < threads; i++) {
        train_thread(i);
      }
    }
    printf("epoch %d %f\n", e, res);
  }

  struct timeval end; gettimeofday(&end, NULL);
  printf("END TRAINING. Duration: %.3fs\n", (end.tv_sec-stt.tv_sec) + (end.tv_usec-stt.tv_usec)/(1e6));
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


// TEST
intT testTripleNum, trainTripleNum, validTripleNum;
intT headType[1000000], tailType[1000000];
intT nnTotal[5];
LabelTriple *tripleList, *testList;

intT l_filter_tot[6], l_filter_rank[6], l_tot[6], l_rank[6];
intT r_filter_tot[6], r_filter_rank[6], r_tot[6], r_rank[6];

void testInit() {
  struct timeval stt; gettimeofday(&stt, NULL);
  printf("START TEST INITIALIZATION ...\n");

  FILE *fin;
  int tmp;

  fin = fopen((inputDir + "relation2id.txt").c_str(), "r");
  tmp = fscanf(fin, "%d", &relationNum);
  fclose(fin);
  
  fin = fopen((inputDir + "entity2id.txt").c_str(), "r");
  tmp = fscanf(fin, "%d", &entityNum);
  fclose(fin);

  vecBuf = (floatT *) malloc((relationNum + entityNum) * dimension * sizeof(floatT));
  rVecBuf = vecBuf;
  eVecBuf = vecBuf + relationNum * dimension;

  headBegins = (intT *) malloc(relationNum * 4 * sizeof(intT));
  headEnds   = headBegins + relationNum;
  tailBegins = headEnds   + relationNum;
  tailEnds   = tailBegins + relationNum;

  FILE* f_kb1 = fopen((inputDir + "test2id_all.txt").c_str(), "r");
  FILE* f_kb2 = fopen((inputDir + "train2id.txt").c_str(), "r");
  FILE* f_kb3 = fopen((inputDir + "valid2id.txt").c_str(), "r");
  tmp = fscanf(f_kb1, "%d", &testTripleNum);
  tmp = fscanf(f_kb2, "%d", &trainTripleNum);
  tmp = fscanf(f_kb3, "%d", &validTripleNum);
  tripleNum = testTripleNum + trainTripleNum + validTripleNum;
  tripleList = (LabelTriple *) malloc(tripleNum * dimension * sizeof(LabelTriple));
  testList = (LabelTriple *) malloc(testTripleNum * dimension * sizeof(LabelTriple));

  intT label, h, t, r;
  for (intT i = 0; i < testTripleNum; i++) {
    tmp = fscanf(f_kb1, "%d %d %d %d", &label, &h, &t, &r);
    label++;
    nnTotal[label]++;
    testList[i].h = h; testList[i].t = t; testList[i].r = r;
    testList[i].label = label;
    tripleList[i].h = h; tripleList[i].t = t; tripleList[i].r = r;
  }
  // memcpy(testList, tripleList, testTripleNum * dimension * sizeof(floatT));

  LabelTriple *tmpTrainList = tripleList + testTripleNum;
  LabelTriple *tmpValidList = tmpTrainList + trainTripleNum;
  for (intT i = 0; i < trainTripleNum; i++) {
    tmp = fscanf(f_kb2, "%d %d %d", &tmpTrainList[i].h, &tmpTrainList[i].t, &tmpTrainList[i].r);
  }
  for (intT i = 0; i < validTripleNum; i++) {
    tmp = fscanf(f_kb2, "%d %d %d", &tmpValidList[i].h, &tmpValidList[i].t, &tmpValidList[i].r);
  }
  fclose(f_kb1);
  fclose(f_kb2);
  fclose(f_kb3);

  sort(tripleList, tripleList + tripleNum, cmp_head());

  intT hTypeOffset = 0, tTypeOffset = 0;
  FILE* f_type = fopen((inputDir + "type_constrain.txt").c_str(),"r");
  tmp = fscanf(f_type, "%d", &tmp);
  for (intT i = 0; i < relationNum; i++) {
    int rel, tot;
    tmp = fscanf(f_type, "%d %d", &rel, &tot);
    headBegins[rel] = hTypeOffset;
    for (int j = 0; j < tot; j++) {
      tmp = fscanf(f_type, "%d", &headType[hTypeOffset]);
      hTypeOffset++;
    }
    headEnds[rel] = hTypeOffset;
    sort(headType + headBegins[rel], headType + headEnds[rel]);

    tmp = fscanf(f_type, "%d %d", &rel, &tot);
    tailBegins[rel] = tTypeOffset;
    for (int j = 0; j < tot; j++) {
      tmp = fscanf(f_type, "%d", &tailType[tTypeOffset]);
      tTypeOffset++;
    }
    tailEnds[rel] = tTypeOffset;
    sort(tailType + tailBegins[rel], tailType + tailEnds[rel]);
  }
  fclose(f_type);

  struct timeval end; gettimeofday(&end, NULL);
  printf("END TEST INITIALIZATION. INIT duration: %.3fs\n", (end.tv_sec-stt.tv_sec) + (end.tv_usec-stt.tv_usec)/(1e6));
}

bool find(intT h, intT t, intT r) {
  intT lef = 0;
  intT rig = tripleNum - 1;
  intT mid;
  while (lef + 1 < rig) {
    intT mid = (lef + rig) >> 1;
    if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) ||
        (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) {
        lef = mid;
    } else {
      rig = mid;
    }
  }
  if ((tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) ||
      (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t)) {
      return true;
  }
  return false;
}

void testMode(int tid) {
  #pragma omp parallel for
  for (intT i = 0; i < testTripleNum; i++) {
    intT h = testList[i].h, t = testList[i].t, r = testList[i].r, label = testList[i].label;
    floatT *hVec = eVecBuf + h * dimension;
    floatT *tVec = eVecBuf + t * dimension;
    floatT *rVec = rVecBuf + r * dimension;
    floatT *jVec;

    floatT minimal = tripleDiff(hVec, tVec, rVec);
    intT l_filter_s = 0; intT l_s = 0; intT l_filter_s_constrain = 0; intT l_s_constrain = 0;
    intT r_filter_s = 0; intT r_s = 0; intT r_filter_s_constrain = 0; intT r_s_constrain = 0;
    intT hType = headBegins[r]; intT tType = tailBegins[r];
    for (intT j = 0; j < entityNum; j++) {
      jVec = eVecBuf + j * dimension;
      if (j != h) {
        float value = tripleDiff(jVec, tVec, rVec);
        if (value < minimal) {
          l_s += 1;
          if (not find(j, t, r))
            l_filter_s += 1;
        }
        while (hType < headEnds[r] && headType[hType] < j) hType++;
        if (hType < headEnds[r] && headType[hType] == j) {
          if (value < minimal) {
            l_s_constrain += 1;
            if (not find(j, t, r))
              l_filter_s_constrain += 1;
          }
        }
      }
      if (j != t) {
        float value = tripleDiff(hVec, jVec, rVec);
        if (value < minimal) {
          r_s += 1;
          if (not find(h, j, r))
            r_filter_s += 1;
        }
        while (tType < tailEnds[r] && tailType[tType] < j) tType++;
        if (tType < tailEnds[r] && tailType[tType] == j) {
          if (value < minimal) {
            r_s_constrain += 1;
            if (not find(h, j, r))
              r_filter_s_constrain += 1;
          }
        }
      }
    }
    if (l_filter_s < 10) { __sync_fetch_and_add(l_filter_tot+0, 1); }
    if (l_s < 10)        { __sync_fetch_and_add(l_tot+0, 1); }
    if (r_filter_s < 10) { __sync_fetch_and_add(r_filter_tot+0, 1); }
    if (r_s < 10)        { __sync_fetch_and_add(r_tot+0, 1); }
    
    __sync_fetch_and_add(l_filter_rank+0, l_filter_s);
    __sync_fetch_and_add(r_filter_rank+0, r_filter_s);
    __sync_fetch_and_add(l_rank+0, l_s);
    __sync_fetch_and_add(r_rank+0, r_s);

    if (l_filter_s < 10) { __sync_fetch_and_add(l_filter_tot+label, 1); }
    if (l_s < 10)        { __sync_fetch_and_add(l_tot+label, 1); }
    if (r_filter_s < 10) { __sync_fetch_and_add(r_filter_tot+label, 1); }
    if (r_s < 10)        { __sync_fetch_and_add(r_tot+label, 1); }
    __sync_fetch_and_add(l_filter_rank+label, l_filter_s);
    __sync_fetch_and_add(r_filter_rank+label, r_filter_s);
    __sync_fetch_and_add(l_rank+label, l_s);
    __sync_fetch_and_add(r_rank+label, r_s); 

    if (l_filter_s_constrain < 10) { __sync_fetch_and_add(l_filter_tot+5, 1); }
    if (l_s_constrain < 10)        { __sync_fetch_and_add(l_tot+5, 1); }
    if (r_filter_s_constrain < 10) { __sync_fetch_and_add(r_filter_tot+5, 1); }
    if (r_s_constrain < 10)        { __sync_fetch_and_add(r_tot+5, 1); }
    __sync_fetch_and_add(l_filter_rank+5, l_filter_s_constrain);
    __sync_fetch_and_add(r_filter_rank+5, r_filter_s_constrain);
    __sync_fetch_and_add(l_rank+5, l_s_constrain);
    __sync_fetch_and_add(r_rank+5, r_s_constrain); 
  }
}

void test() {
  struct timeval stt; gettimeofday(&stt, NULL);
  printf("START TESTING ...\n");

  omp_set_num_threads(threads);
  testMode(0);
  for (int i = 0; i <= 0; i++) {
    printf("left %f %f\n", 1.0*l_rank[i] / testTripleNum, 1.0*l_tot[i] / testTripleNum);
    printf("left(filter) %f %f\n", 1.0*l_filter_rank[i] / testTripleNum, 1.0*l_filter_tot[i] / testTripleNum);
    printf("right %f %f\n", 1.0*r_rank[i] / testTripleNum, 1.0*r_tot[i] / testTripleNum);
    printf("right(filter) %f %f\n", 1.0*r_filter_rank[i] / testTripleNum, 1.0*r_filter_tot[i] / testTripleNum);
	}
  for (int i = 5; i <= 5; i++) {
    printf("left %f %f\n", 1.0*l_rank[i] / testTripleNum, 1.0*l_tot[i] / testTripleNum);
    printf("left(filter) %f %f\n", 1.0*l_filter_rank[i] / testTripleNum, 1.0*l_filter_tot[i] / testTripleNum);
    printf("right %f %f\n", 1.0*r_rank[i] / testTripleNum, 1.0*r_tot[i] / testTripleNum);
    printf("right(filter) %f %f\n", 1.0*r_filter_rank[i] / testTripleNum, 1.0*r_filter_tot[i] / testTripleNum);
  }
	for (int i = 1; i <= 4; i++) {
    printf("left %f %f\n", 1.0*l_rank[i] / nnTotal[i], 1.0*l_tot[i] / nnTotal[i]);
    printf("left(filter) %f %f\n", 1.0*l_filter_rank[i] / nnTotal[i], 1.0*l_filter_tot[i] / nnTotal[i]);
    printf("right %f %f\n", 1.0*r_rank[i] / nnTotal[i], 1.0*r_tot[i] / nnTotal[i]);
    printf("right(filter) %f %f\n", 1.0*r_filter_rank[i] / nnTotal[i], 1.0*r_filter_tot[i] / nnTotal[i]);
	}

  struct timeval end; gettimeofday(&end, NULL);
  printf("END TESTING. TEST duration: %.3fs\n", (end.tv_sec-stt.tv_sec) + (end.tv_usec-stt.tv_usec)/(1e6));
}

void testFinish() {
  free(vecBuf);
  free(headBegins);
  free(tripleList);
}

#endif // !PARTRANSE_Hd
