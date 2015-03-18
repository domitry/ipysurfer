typedef struct __attribute__ ((packed)){
  int version;
  int width;
  int height;
  int depth;
  int nframes;
  int type;
  int dof;
  short goodRASFlag;
  float delta[3];
  float Mdc[9];
  float Pxyz_c[3];
  unsigned char padding[256-4*15+2];
}MGH_HEADER;

typedef unsigned char uchar;

#define MRI_UCHAR    0
#define MRI_INT      1
#define MRI_LONG     2
#define MRI_FLOAT    3
#define MRI_SHORT    4
#define MRI_BITMAP   5
