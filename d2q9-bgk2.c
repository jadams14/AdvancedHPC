/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>


#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
/*typedef struct
{
  float speeds[NSPEEDS];
} t_speed;*/

typedef struct {
  float* speeds;
  float* speeds1;
  float* speeds2;
  float* speeds3;
  float* speeds4;
  float* speeds5;
  float* speeds6;
  float* speeds7;
  float* speeds8;
} t_speed;
/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* obstacles);
int accelerate_flow(const t_param  params, t_speed* restrict cells, int* restrict obstacles);
int propagate(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells);
int rebound(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles);
float collision(const t_param params, const t_speed* restrict cells, t_speed* restrict tmp_cells, const int* restrict obstacles);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* restrict cells,  int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);
/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &obstacles, &av_vels);

  cells = (t_speed*)_mm_malloc(sizeof(t_speed), 64);
  tmp_cells = (t_speed*)_mm_malloc(sizeof(t_speed), 64);
  cells->speeds = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds1 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds2 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds3 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds4 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds5 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds6 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds7 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  cells->speeds8 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds1 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds2 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds3 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds4 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds5 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds6 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds7 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  tmp_cells->speeds8 = (float*)_mm_malloc(sizeof(float) * (params.ny * params.nx), 64);
  
  
  /* initialise densities */
  const float w0 = params.density * 4.f / 9.f;
  const float w1 = params.density      / 9.f;
  const float w2 = params.density      / 36.f;

  //OMP_NUM_THREADS = 16 
  //OMP_PROC_BIND = true
  #pragma omp parallel for
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* centre */
      cells->speeds[ii + jj*params.nx] = w0;
      /* axis directions */
      cells->speeds1[ii + jj*params.nx] = w1;
      cells->speeds2[ii + jj*params.nx] = w1;
      cells->speeds3[ii + jj*params.nx] = w1;
      cells->speeds4[ii + jj*params.nx] = w1;
      /* diagonals */
      cells->speeds5[ii + jj*params.nx] = w2;
      cells->speeds6[ii + jj*params.nx] = w2;
      cells->speeds7[ii + jj*params.nx] = w2;
      cells->speeds8[ii + jj*params.nx] = w2;
    }
  }

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  //#pragma omp parallel for
  for (int tt = 0; tt < params.maxIters; tt+=2)
  {    
    av_vels[tt] = timestep(params, cells, tmp_cells, obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
    av_vels[tt+1] = timestep(params, tmp_cells, cells, obstacles);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt+1);
    printf("av velocity: %.12E\n", av_vels[tt+1]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* write final values and free memory */
  printf("==done==\n");
  printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
  printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  write_values(params, cells, obstacles, av_vels);
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  return EXIT_SUCCESS;
}

float timestep(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles)
{
  accelerate_flow(params, cells, obstacles);
  float average = collision(params, cells, tmp_cells, obstacles);
  return average;
}

int accelerate_flow(const t_param params, t_speed* restrict cells, int* restrict obstacles)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int jj = params.ny - 2;

  for (int ii = 0; ii < params.nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*params.nx]
        && (cells->speeds3[ii + jj*params.nx] - w1) > 0.f
        && (cells->speeds6[ii + jj*params.nx] - w2) > 0.f
        && (cells->speeds7[ii + jj*params.nx] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speeds1[ii + jj*params.nx] += w1;
      cells->speeds5[ii + jj*params.nx] += w2;
      cells->speeds8[ii + jj*params.nx] += w2;
      /* decrease 'west-side' densities */
      cells->speeds3[ii + jj*params.nx] -= w1;
      cells->speeds6[ii + jj*params.nx] -= w2;
      cells->speeds7[ii + jj*params.nx] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells)
{
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? ( params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? ( params.nx - 1) : (ii - 1);
      tmp_cells->speeds[ii + jj*params.nx]  = cells->speeds[ii + jj*params.nx]; /* central cell, no movement */
      tmp_cells->speeds1[ii + jj*params.nx] = cells->speeds1[x_w + jj*params.nx]; /* east */
      tmp_cells->speeds2[ii + jj*params.nx] = cells->speeds2[ii + y_s*params.nx]; /* north */
      tmp_cells->speeds3[ii + jj*params.nx] = cells->speeds3[x_e + jj*params.nx]; /* west */
      tmp_cells->speeds4[ii + jj*params.nx] = cells->speeds4[ii + y_n*params.nx]; /* south */
      tmp_cells->speeds5[ii + jj*params.nx] = cells->speeds5[x_w + y_s*params.nx]; /* north-east */
      tmp_cells->speeds6[ii + jj*params.nx] = cells->speeds6[x_e + y_s*params.nx]; /* north-west */
      tmp_cells->speeds7[ii + jj*params.nx] = cells->speeds7[x_e + y_n*params.nx]; /* south-west */
      tmp_cells->speeds8[ii + jj*params.nx] = cells->speeds8[x_w + y_n*params.nx]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

int rebound(const t_param params, t_speed* restrict cells, t_speed* restrict tmp_cells, int* restrict obstacles)
{
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    { 
      if (obstacles[jj*params.nx + ii])
      {
        cells->speeds1[jj*params.nx + ii] = tmp_cells->speeds3[jj*params.nx + ii];
        cells->speeds2[jj*params.nx + ii] = tmp_cells->speeds4[jj*params.nx + ii];
        cells->speeds3[jj*params.nx + ii] = tmp_cells->speeds1[jj*params.nx + ii];
        cells->speeds4[jj*params.nx + ii] = tmp_cells->speeds2[jj*params.nx + ii];
        cells->speeds5[jj*params.nx + ii] = tmp_cells->speeds7[jj*params.nx + ii];
        cells->speeds6[jj*params.nx + ii] = tmp_cells->speeds8[jj*params.nx + ii];
        cells->speeds7[jj*params.nx + ii] = tmp_cells->speeds5[jj*params.nx + ii];
        cells->speeds8[jj*params.nx + ii] = tmp_cells->speeds6[jj*params.nx + ii];
      }
    }
  }

  return EXIT_SUCCESS;
}

float collision(const t_param params, const t_speed* restrict cells, t_speed* restrict tmp_cells, const int* restrict obstacles)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */
  float tot_u = 0.f;
  int tot_cells = 0;
  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */

  //OMP_NUM_THREADS = 16
  //OMP_PLACES=thread
  //OMP_PROC_BIND = true
  //OMP_PLACES=threads
  #pragma omp parallel for reduction(+: tot_u) reduction(+ : tot_cells)
  for (int jj = 0; jj < params.ny; jj++)
  {
    
    //_assume((params.ny)%2==0);
    #pragma omp simd 
    for (int ii = 0; ii < params.nx; ii++)
    {
      __assume_aligned(cells, 64);
      __assume_aligned(tmp_cells, 64);
      __assume_aligned(obstacles, 64);
      const int y_n = (jj + 1) % params.ny;
      const int x_e = (ii + 1) % params.nx;
      const int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      const int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      const int index = ii+(jj*params.nx);
      const float speed = cells->speeds[ii + jj*params.nx];
      const float speed1 = cells->speeds1[x_w + jj*params.nx];
      const float speed2 = cells->speeds2[ii + y_s*params.nx];
      const float speed3 = cells->speeds3[x_e + jj*params.nx];
      const float speed4 = cells->speeds4[ii + y_n*params.nx];
      const float speed5 = cells->speeds5[x_w + y_s*params.nx];
      const float speed6 = cells->speeds6[x_e + y_s*params.nx];
      const float speed7 = cells->speeds7[x_e + y_n*params.nx];
      const float speed8 = cells->speeds8[x_w + y_n*params.nx];
      /*if (obstacles[index]) {
        tmp_cells->speeds1[index] = speed3; 
        tmp_cells->speeds2[index] = speed4;
        tmp_cells->speeds3[index] = speed1;
        tmp_cells->speeds4[index] = speed2;
        tmp_cells->speeds5[index] = speed7;
        tmp_cells->speeds6[index] = speed8;
        tmp_cells->speeds7[index] = speed5;
        tmp_cells->speeds8[index] = speed6;
      }
      else {
        tmp_cells->speeds[index] = speed;        
        tmp_cells->speeds1[index] = speed1;
        tmp_cells->speeds2[index] = speed2;
        tmp_cells->speeds3[index] = speed3;
        tmp_cells->speeds4[index] = speed4;
        tmp_cells->speeds5[index] = speed5;
        tmp_cells->speeds6[index] = speed6;
        tmp_cells->speeds7[index] = speed7;
        tmp_cells->speeds8[index] = speed8;*/
        /* compute local density total */
        const float local_density = speed + speed1 + speed2 + speed3 + speed4 + speed5 + speed6 + speed7 + speed8;

        /* compute x velocity component */
        const float u_x = (speed1
                      + speed5
                      + speed8
                      - (speed3
                         + speed6
                         + speed7))
                     / local_density;
        /* compute y velocity component */
        const float u_y = (speed2
                      + speed5
                      + speed6
                      - (speed4
                         + speed7
                         + speed8))
                     / local_density;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        const float localDensityW1 = w1 * local_density;
        const float localDensityW2 = w2 * local_density;
        const float timesedC_SQ    = (2.f * c_sq * c_sq);
        const float minusU_SQ      = u_sq / (2.f * c_sq);

        const float d_equ = (w0 * local_density
                          * (1.f - minusU_SQ));
        const float d_equ1 = (localDensityW1 * (1.f + u[1] / c_sq
                                     + (u[1] * u[1]) / (timesedC_SQ)
                                     - minusU_SQ));
        const float d_equ2 = (localDensityW1 * (1.f + u[2] / c_sq
                                     + (u[2] * u[2]) / (timesedC_SQ)
                                     - minusU_SQ));
        const float d_equ3 = (localDensityW1 * (1.f + u[3] / c_sq
                                     + (u[3] * u[3]) / (timesedC_SQ)
                                     - minusU_SQ));
        const float d_equ4 = (localDensityW1 * (1.f + u[4] / c_sq
                                     + (u[4] * u[4]) / (timesedC_SQ)
                                     - minusU_SQ));
        const float d_equ5 = (localDensityW2 * (1.f + u[5] / c_sq
                                     + (u[5] * u[5]) / (timesedC_SQ)
                                     - minusU_SQ));
        const float d_equ6 = (localDensityW2 * (1.f + u[6] / c_sq
                                     + (u[6] * u[6]) / (timesedC_SQ)
                                     - minusU_SQ));
        const float d_equ7 = (localDensityW2 * (1.f + u[7] / c_sq
                                   + (u[7] * u[7]) / (timesedC_SQ)
                                     - minusU_SQ));
        const float d_equ8 = (localDensityW2 * (1.f + u[8] / c_sq
                                     + (u[8] * u[8]) / (timesedC_SQ)
                                     - minusU_SQ));

        /* equilibrium densities */
        tmp_cells->speeds[index]  = obstacles[index] ? speed  : (speed + params.omega * (d_equ - speed));
        tmp_cells->speeds1[index] = obstacles[index] ? speed3 : (speed1 + params.omega * (d_equ1 - speed1));
        tmp_cells->speeds2[index] = obstacles[index] ? speed4 : (speed2 + params.omega * (d_equ2 - speed2));
        tmp_cells->speeds3[index] = obstacles[index] ? speed1 : (speed3 + params.omega * (d_equ3 - speed3));
        tmp_cells->speeds4[index] = obstacles[index] ? speed2 : (speed4 + params.omega * (d_equ4 - speed4));
        tmp_cells->speeds5[index] = obstacles[index] ? speed7 : (speed5 + params.omega * (d_equ5 - speed5));
        tmp_cells->speeds6[index] = obstacles[index] ? speed8 : (speed6 + params.omega * (d_equ6 - speed6));
        tmp_cells->speeds7[index] = obstacles[index] ? speed5 : (speed7 + params.omega * (d_equ7 - speed7));
        tmp_cells->speeds8[index] = obstacles[index] ? speed6 : (speed8 + params.omega * (d_equ8 - speed8));

        /* accumulate the norm of x- and y- velocity components */
        tot_u += obstacles[index] ? 0 : sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        tot_cells += obstacles[index] ? 0 : 1;
      //}
      //tot_u = tot_u;
      //tot_cells = tot_cells;
    }
  }
  return tot_u / (float) tot_cells;
}

float av_velocity(const t_param params, t_speed* restrict cells, int* obstacles)
{
  //int    tot_cells = 0;  // no. of cells used in calculation 
  //float tot_u;          // accumulated magnitudes of velocity for each cell 
  int tot_cells = 0;
  float tot_u = 0.f;

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int index = ii + ( jj * params.nx);
      // ignore occupied cells 
      if (!obstacles[index])
      {
        // local density total 
        float local_density = 0.f;

        local_density = cells->speeds[index] 
                      + cells->speeds1[index] 
                      + cells->speeds2[index] 
                      + cells->speeds3[index] 
                      + cells->speeds4[index] 
                      + cells->speeds5[index] 
                      + cells->speeds6[index] 
                      + cells->speeds7[index] 
                      + cells->speeds8[index];

        // x-component of velocity 
        float u_x = (cells->speeds1[index]
                      + cells->speeds5[index]
                      + cells->speeds8[index]
                      - (cells->speeds3[index]
                         + cells->speeds6[index]
                         + cells->speeds7[index]))
                     / local_density;
        // compute y velocity component 
        float u_y = (cells->speeds2[index]
                      + cells->speeds5[index]
                      + cells->speeds6[index]
                      - (cells->speeds4[index]
                         + cells->speeds7[index]
                         + cells->speeds8[index]))
                     / local_density;
        
        // accumulate the norm of x- and y- velocity components 
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        // increase counter of inspected cells 
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  
  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);


  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  _mm_free(*cells_ptr);
  *cells_ptr = NULL;

  _mm_free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      int index = ii + (jj*params.nx);
      total += cells->speeds[index] 
                      + cells->speeds1[index] 
                      + cells->speeds2[index] 
                      + cells->speeds3[index] 
                      + cells->speeds4[index] 
                      + cells->speeds5[index] 
                      + cells->speeds6[index] 
                      + cells->speeds7[index] 
                      + cells->speeds8[index];
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      const int index = ii + jj*params.nx;
      /* an occupied cell */
      if (obstacles[index])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;
        local_density = cells->speeds[index] 
                      + cells->speeds1[index] 
                      + cells->speeds2[index] 
                      + cells->speeds3[index] 
                      + cells->speeds4[index] 
                      + cells->speeds5[index] 
                      + cells->speeds6[index] 
                      + cells->speeds7[index] 
                      + cells->speeds8[index];

        /* compute x velocity component */
        u_x = (cells->speeds1[index]
               + cells->speeds5[index]
               + cells->speeds8[index]
               - (cells->speeds3[index]
                  + cells->speeds6[index]
                  + cells->speeds7[index]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells->speeds2[index]
               + cells->speeds5[index]
               + cells->speeds6[index]
               - (cells->speeds4[index]
                  + cells->speeds7[index]
                  + cells->speeds8[index]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}