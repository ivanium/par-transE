#ifndef PARSECMDLINE_H
#define PARSECMDLINE_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>

#include "util.h"
#include "transE.h"

// Hyperparameters
extern floatT alpha;
extern floatT margin; 
extern intT dimension;
extern intT bernFlag;
extern intT epochs;
extern intT nbatches;

// Arguments
extern intT loadBinaryFlag;
extern intT outBinaryFlag;
extern std::string inputDir;
extern std::string outputDir;
extern std::string loadDir;
extern std::string note;

static const char *optString = "s:i:o:Bl:bN:e:n:a:m:vh?";

static const struct option longOpts[] = {
    {"size", required_argument, NULL, 's'},
    {"input", required_argument, NULL, 'i'},
    {"output", required_argument, NULL, 'o'},
    {"output-bin", no_argument, NULL, 'B'},
    {"load", required_argument, NULL, 'l'},
    {"load-bin", no_argument, NULL, 'b'},
    {"note", required_argument, NULL, 'N'},
    {"threads", required_argument, NULL, 't'},
    {"epochs", required_argument, NULL, 'e'},
    {"nbatchs", required_argument, NULL, 'n'},
    {"alpha", required_argument, NULL, 'a'},
    {"margin", required_argument, NULL, 'm'},
    {"verbose", no_argument, NULL, 'v'},
    {"help", no_argument, NULL, 'h'},
    {NULL, no_argument, NULL, 0}};

/* Display program usage, and exit.
 */
void display_usage(void) {
  puts("transE - Knowledge Embedding");
  puts("./transX [-size SIZE] [-sizeR SIZER]");
  puts("         [-input INPUT] [-output OUTPUT] [-load LOAD]");
  puts("         [-load-binary 0/1] [-out-binary 0/1]");
  puts("         [-thread THREAD] [-epochs EPOCHS] [-nbatches NBATCHES]");
  puts("         [-alpha ALPHA] [-margin MARGIN]");
  puts("         [-note NOTE]");
  puts("");
  puts("optional arguments:");
  puts("--size SIZE           dimension of entity embeddings");
  puts("--sizeR SIZER         dimension of relation embeddings");
  puts("--input INPUT         folder of training data");
  puts("--output OUTPUT       folder of outputing results");
  puts("--load LOAD           folder of pretrained data");
  puts("--load-binary [0/1]   [1] pretrained data need to load in is in the binary form");
  puts("--out-binary [0/1]    [1] results will be outputed in the binary form");
  puts("--thread THREAD       number of worker threads");
  puts("--epochs EPOCHS       number of epochs");
  puts("--nbatches NBATCHES   number of batches for each epoch");
  puts("--alpha ALPHA         learning rate");
  puts("--margin MARGIN       margin in max-margin loss for pairwise training");
  puts("--note NOTE           information you want to add to the filename");
  exit(EXIT_FAILURE);
}

void show_args() {
  printf("----------ARGUMENTS----------\n");
  printf("size: %d\n", dimension);
  printf("input: %s\n", inputDir.c_str());
  printf("output: %s\n", outputDir.c_str());
  printf("output-bin: %d\n", outBinaryFlag);
  printf("load: %s\n", loadDir.c_str());
  printf("load-bin: %d\n", loadBinaryFlag);
  printf("note: %s\n", note.c_str());
  printf("epochs: %d\n", epochs);
  printf("nbatchs: %d\n", nbatches);
  printf("alpha: %.3f\n", alpha);
  printf("margin: %.3f\n", margin);
  printf("-----------------------------\n");
}

#define parseCmdArgs(argc, argv) {                                \
  int opt = 0;                                                    \
  int longIndex = 0;                                              \
  bool verbose = false;                                           \
                                                                  \
  /* Initialize global Args before we get to work. */             \
                                                                  \
  opt = getopt_long(argc, argv, optString, longOpts, &longIndex); \
  while (opt != -1) {                                             \
    switch (opt) {                                                \
    case 's':                                                     \
      dimension = atoi(optarg);                                   \
      break;                                                      \
    case 'i':                                                     \
      inputDir = optarg;                                          \
      break;                                                      \
    case 'o':                                                     \
      outputDir = optarg;                                         \
      break;                                                      \
    case 'B':                                                     \
      outBinaryFlag = 1;                                          \
      break;                                                      \
    case 'l':                                                     \
      loadDir = optarg;                                           \
      break;                                                      \
    case 'b':                                                     \
      loadBinaryFlag = 1;                                         \
      break;                                                      \
    case 'N':                                                     \
      note = optarg;                                              \
      break;                                                      \
    case 'e':                                                     \
      epochs = atoi(optarg);                                      \
      break;                                                      \
    case 'n':                                                     \
      nbatches = atoi(optarg);                                    \
      break;                                                      \
    case 'a':                                                     \
      alpha = atoi(optarg);                                       \
      break;                                                      \
    case 'm':                                                     \
      margin = atoi(optarg);                                      \
      break;                                                      \
    case 'v':                                                     \
      verbose = true;                                             \
      break;                                                      \
    case 'h': /* fall-through is intentional */                   \
    case '?':                                                     \
      display_usage();                                            \
      break;                                                      \
    default:                                                      \
      /* Won't actually get here. */                              \
      break;                                                      \
    }                                                             \
    opt=getopt_long(argc, argv, optString, longOpts, &longIndex); \
  }                                                               \
  if (verbose) show_args();                                       \
}

int main(int argc, char *argv[]) {
  parseCmdArgs(argc, argv);
  testInit();
  load();
  test();
  testFinish();
  return 0;
}

#endif // !PARSECMDLINE_H
