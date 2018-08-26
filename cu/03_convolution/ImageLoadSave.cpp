#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ImageLoadSave.h"

void read_image_header( string path, int* width, int* height )
{

	FILE* pFile;

	pFile = fopen( path.c_str(), "r" );

    int depth;
    char magicNumber[3];

    int err = fscanf(pFile, "%s", magicNumber);


    if	( ( strcmp ( magicNumber, "P5") != 0 ) &&
	  ( strcmp ( magicNumber, "P2") != 0 ) )
    {
    }
    
    err = fscanf (pFile, "%d %d %d", width, height, &depth );
//	printf ("width: %d\t height:%d\t depth:%d\n", *width, *height, depth );

    fclose( pFile ); 

}

PPM read_ppm( string path )
{

	FILE* pFile;
	PPM ppm;

	pFile = fopen( path.c_str(), "r" );

    int depth;
    char magicNumber[3];

    int err = fscanf(pFile, "%s", magicNumber);

    if	( ( strcmp ( magicNumber, "P5") != 0 ) &&
	  ( strcmp ( magicNumber, "P2") != 0 ) )
    {
//	return NULL;
    }
    
    err = fscanf (pFile, "%d %d %d", &ppm.width, &ppm.height, &depth );
    ppm.value = (unsigned char *) malloc( sizeof(unsigned char) * (ppm.width) * (ppm.height) * 3) ;
//    ppm.value = ppmvalue;
	err = (int)fread( ppm.value, sizeof(uchar), ppm.width * ppm.height * 3, pFile);

    fclose( pFile ); 

    return ppm;
}

PGM read_pgm( string path )
{
	FILE* pFile;
	PGM pgm;

	pFile = fopen( path.c_str(), "r" );
//	cout << "filename: " << path << endl;

    int depth;
    char magicNumber[3];

    int err =fscanf(pFile, "%s", magicNumber);

    if	( strcmp(magicNumber, "P5") && strcmp(magicNumber, "P2") )
    {
//		return NULL;
    }
    
    err = fscanf (pFile, "%d %d %d", &pgm.width, &pgm.height, &depth );

    pgm.value = (unsigned char *) malloc( sizeof(unsigned char) * (pgm.width) * (pgm.height) ) ;
	err = (int)fread( pgm.value, sizeof(unsigned char), pgm.width * pgm.height, pFile);

    fclose( pFile ); 

    return pgm;
}

void write_pgm(PGM pgm, string filename)
{
 
    FILE* pFile; 
 
    pFile = fopen (filename.c_str(), "wb" );

    fprintf(pFile, "P5\n");
    fprintf(pFile, "%d %d\n255\n", pgm.width, pgm.height );
   
    fwrite( (void*)pgm.value, sizeof(unsigned char), pgm.width * pgm.height, pFile);
      
    fclose(pFile); 
}


void write_ppm(PPM ppm, string filename)
{
 
    FILE* pFile; 
 
    pFile = fopen (filename.c_str(), "wb" );

    fprintf(pFile, "P6\n");
    fprintf(pFile, "%d %d\n255", ppm.width, ppm.height );
  
    fwrite( (void*)ppm.value, sizeof(unsigned char), ppm.width * ppm.height * 3, pFile);
      
    fclose(pFile); 
}

void expand_ppm(PPM orig_ppm, string filename, int newwidth, int newheight)
{
	PPM ppm;
	ppm.width = newwidth;
	ppm.height = newheight;
	ppm.value = (uchar*) malloc( sizeof(uchar) * ppm.width * ppm.height * 3 );

	for(int y=0; y<ppm.height; y++) {
		for(int x=0; x<ppm.width*3; x++) {
			ppm.value[y * ppm.width*3 + x  ] = orig_ppm.value[(y%orig_ppm.height) * orig_ppm.width*3 + (x%(orig_ppm.width*3))  ];
		}
	}

	FILE* pFile; 
 
    pFile = fopen (filename.c_str(), "wb" );

    fprintf(pFile, "P6\n");
    fprintf(pFile, "%d %d\n255", ppm.width, ppm.height );
    
	fwrite( (void*)ppm.value, sizeof(unsigned char), ppm.width * ppm.height * 3, pFile);

    fclose(pFile); 
}
