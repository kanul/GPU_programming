#ifndef IMAGE_LOAD_SAVE_H
#define IMAGE_LOAD_SAVE_H

#include <iostream>
#include <string.h>

using namespace std;

typedef unsigned char uchar;

typedef struct _PPM
{
	int width;
	int height;
	uchar* value;
} PPM;

typedef struct _PGM {
	int width;
	int height;
	unsigned char* value;
} PGM;

void read_image_header( string path, int* width, int* height );
PPM read_ppm( string path);
void read_ppm_map( string path, PPM *ppm);
PGM read_pgm( string path );

void write_pgm(PGM pgm, string filename);
void write_ppm(PPM pgm, string filename);

void expand_ppm(PPM orig_ppm, string filename, int newwidth, int newheight);

#endif
