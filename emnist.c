#include <arpa/inet.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef enum {
  INVALID = 0,
  UCHAR,
  SCHAR,
  SHORT,
  INT,
  FLOAT,
  DOUBLE,
} CType;

CType type_from_magic(int magic) {
  switch (magic) {
  case 0x08:
    return UCHAR; // unsigned byte
  case 0x09:
    return SCHAR; // signed byte
  case 0x0B:
    return SHORT; // short (2 bytes)
  case 0x0C:
    return INT; // int (4 bytes)
  case 0x0D:
    return FLOAT; // float (4 bytes)
  case 0x0E:
    return DOUBLE; // double (8 bytes)

  default:
    return INVALID;
  }
}

unsigned short ctype_size(CType type) {
  switch (type) {
  case UCHAR:
    return 1; // unsigned byte
  case SCHAR:
    return 1; // signed byte
  case SHORT:
    return 2; // short (2 bytes)
  case INT:
    return 4; // int (4 bytes)
  case FLOAT:
    return 4; // float (4 bytes)
  case DOUBLE:
    return 8; // double (8 bytes)

  default:
    return 0;
  }
}

#define MAX_DIM 255

#include "nn.c"

int main(void) {
  CType element_type;
  unsigned short dim;
  if (!(fgetc(stdin) == 0 && fgetc(stdin) == 0) ||
      (element_type = type_from_magic(fgetc(stdin))) == 0 ||
      (dim = (unsigned short)fgetc(stdin)) == 0) {
    fprintf(stderr, "magic number wrong, the file is not an idx file\n");
    return 1;
  }

  unsigned dim_sizes[MAX_DIM] = {0};
  unsigned network_input = 1;

  unsigned data_length = 0;

  for (unsigned i = 0; i < dim; i++) {
    if (1 != fread(dim_sizes + i, 4, 1, stdin)) {
      fprintf(stderr, "error reading idx file sizes\n");
      return 1;
    }
    dim_sizes[i] = ntohl(dim_sizes[i]);
    if (i > 0) {
      network_input *= dim_sizes[i];
      data_length *= dim_sizes[i];
    } else {
      data_length += dim_sizes[i];
    }
    printf("dim size %u = %u\n", i, dim_sizes[i]);
  }

  void *data = malloc(data_length * ctype_size(element_type));

  if (data_length !=
      fread(data, ctype_size(element_type), data_length, stdin)) {
    fprintf(stderr, "failed reading idx file\n");
    return 1;
  }

  NNNetwork network = {0};
  init_network(&network, 4, (unsigned[]){network_input, 2, 2, 2});
  init_network_weights(&network);
  NNTrainer trainer = {0};
  init_trainer(&trainer, &network);

  batch_backpropagation_step(&trainer, 4, data, data);
  apply_slopes(&trainer, 0.1f);

  free_network(&network);
  free_trainer(&trainer);
  free(data);
  return 0;
}
