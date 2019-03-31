#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9


typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  const int totalSize = 0;
  
  for (int ii = 0; ii < nx; ii++)
  {
    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj*nx]
        && (cells->speeds[ii + jj*nx + (3 * totalSize)] - w1) > 0.f
        && (cells->speeds[ii + jj*nx + (6 * totalSize)] - w2) > 0.f
        && (cells->speeds[ii + jj*nx + (7 * totalSize)] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells->speeds[ii + jj*nx + (1 * totalSize)] += w1;
      cells->speeds[ii + jj*nx + (5 * totalSize)] += w2;
      cells->speeds[ii + jj*nx + (8 * totalSize)] += w2;
      /* decrease 'west-side' densities */
      cells->speeds[ii + jj*nx + (3 * totalSize)] -= w1;
      cells->speeds[ii + jj*nx + (6 * totalSize)] -= w2;
      cells->speeds[ii + jj*nx + (7 * totalSize)] -= w2;
    }
  }
}

kernel void rebound(global t_speed* cells,
                    global t_speed* tmp_cells,
                    global int* obstacles,
                    int nx, int ny) {
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  
  if (obstacles[jj * nx + ii]) {

    /* called after propagate, so taking values from scratch space
    ** mirroring, and writing into main grid */
    cells[ii + jj * nx].speeds[1] = tmp_cells[ii + jj * nx].speeds[3];
    cells[ii + jj * nx].speeds[2] = tmp_cells[ii + jj * nx].speeds[4];
    cells[ii + jj * nx].speeds[3] = tmp_cells[ii + jj * nx].speeds[1];
    cells[ii + jj * nx].speeds[4] = tmp_cells[ii + jj * nx].speeds[2];
    cells[ii + jj * nx].speeds[5] = tmp_cells[ii + jj * nx].speeds[7];
    cells[ii + jj * nx].speeds[6] = tmp_cells[ii + jj * nx].speeds[8];
    cells[ii + jj * nx].speeds[7] = tmp_cells[ii + jj * nx].speeds[5];
    cells[ii + jj * nx].speeds[8] = tmp_cells[ii + jj * nx].speeds[6];
  }
}

kernel void propagate(global t_speed* cells,
                      global t_speed* tmp_cells,
                      global int* obstacles,
                      int nx, int ny) {
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);

  /* determine indices of axis-direction neighbours
  ** respecting periodic boundary conditions (wrap around) */
  int y_n = (jj + 1) % ny;
  int x_e = (ii + 1) % nx;
  int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  /* propagate densities from neighbouring cells, following
  ** appropriate directions of travel and writing into
  ** scratch space grid */
  tmp_cells[ii + jj*nx].speeds[0] = cells[ii + jj*nx].speeds[0]; /* central cell, no movement */
  tmp_cells[ii + jj*nx].speeds[1] = cells[x_w + jj*nx].speeds[1]; /* east */
  tmp_cells[ii + jj*nx].speeds[2] = cells[ii + y_s*nx].speeds[2]; /* north */
  tmp_cells[ii + jj*nx].speeds[3] = cells[x_e + jj*nx].speeds[3]; /* west */
  tmp_cells[ii + jj*nx].speeds[4] = cells[ii + y_n*nx].speeds[4]; /* south */
  tmp_cells[ii + jj*nx].speeds[5] = cells[x_w + y_s*nx].speeds[5]; /* north-east */
  tmp_cells[ii + jj*nx].speeds[6] = cells[x_e + y_s*nx].speeds[6]; /* north-west */
  tmp_cells[ii + jj*nx].speeds[7] = cells[x_e + y_n*nx].speeds[7]; /* south-west */
  tmp_cells[ii + jj*nx].speeds[8] = cells[x_w + y_n*nx].speeds[8]; /* south-east */
}

kernel void collision(global t_speed *cells, 
                      global t_speed *tmp_cells, 
                      global int *obstacles, 
                      global float *g_tot_u,
                      local float *l_tot_u,
                      int nx, int ny,
                      float omega) {

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int currentIter = get_global_id(2);
  int g_size_ii = get_global_size(0);
  int g_size_jj = get_global_size(1);
  int l_x = get_local_id(0);
  int l_y = get_local_id(1);

  int l_size_x = get_local_size(0);
  int l_size_y = get_local_size(1);

  int work_group_row = (g_size_ii / l_size_x);
  int work_group_column = (g_size_jj / l_size_y);
  int num_workgroup = work_group_column * work_group_row;


  const int y_n = (jj + 1) % ny;
  const int x_e = (ii + 1) % nx;
  const int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
  const int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

  //Propogate
  const int totalSize = nx * ny;
  const int index = ii+(jj*nx);
  const float speed = cells->speeds[ii + jj*nx + (0 * totalSize)];
  const float speed1 = cells->speeds[x_w + jj*nx + (1 * totalSize)];
  const float speed2 = cells->speeds[ii + y_s*nx + (2 * totalSize)];
  const float speed3 = cells->speeds[x_e + jj*nx + (3 * totalSize)];
  const float speed4 = cells->speeds[ii + y_n*nx + (4 * totalSize)];
  const float speed5 = cells->speeds[x_w + y_s*nx + (5 * totalSize)];
  const float speed6 = cells->speeds[x_e + y_s*nx + (6 * totalSize)];
  const float speed7 = cells->speeds[x_e + y_n*nx + (7 * totalSize)];
  const float speed8 = cells->speeds[x_w + y_n*nx + (8 * totalSize)];

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  


  // if (obstacles[jj * nx + ii]) {
  //   /* called after propagate, so taking values from scratch space
  //   ** mirroring, and writing into main grid */
  //   cells[ii + jj * nx].speeds[1] = tmp_cells[ii + jj * nx].speeds[3];
  //   cells[ii + jj * nx].speeds[2] = tmp_cells[ii + jj * nx].speeds[4];
  //   cells[ii + jj * nx].speeds[3] = tmp_cells[ii + jj * nx].speeds[1];
  //   cells[ii + jj * nx].speeds[4] = tmp_cells[ii + jj * nx].speeds[2];
  //   cells[ii + jj * nx].speeds[5] = tmp_cells[ii + jj * nx].speeds[7];
  //   cells[ii + jj * nx].speeds[6] = tmp_cells[ii + jj * nx].speeds[8];
  //   cells[ii + jj * nx].speeds[7] = tmp_cells[ii + jj * nx].speeds[5];
  //   cells[ii + jj * nx].speeds[8] = tmp_cells[ii + jj * nx].speeds[6];
  // }
  /* don't consider occupied cells */
  // else if (!obstacles[ii + jj * nx])
  // {
  /* compute local density total */
  float local_density = speed + speed1 + speed2 + speed3 + speed4
                        + speed5 + speed6 + speed7 + speed8;
   
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
  float u_sq = u_x * u_x + u_y * u_y;

    /* directional velocity components */
  float u[NSPEEDS];
  u[1] = u_x;        /* east */
  u[2] = u_y;        /* north */
  u[3] = -u_x;       /* west */
  u[4] = -u_y;       /* south */
  u[5] = u_x + u_y;  /* north-east */
  u[6] = -u_x + u_y; /* north-west */
  u[7] = -u_x - u_y; /* south-west */
  u[8] = u_x - u_y;  /* south-east */

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
    /* relaxation step */
  tmp_cells->speeds[index + (0 * totalSize)]  = obstacles[index] ? speed  : (speed + omega * (d_equ - speed));
  tmp_cells->speeds[index + (1 * totalSize)] = obstacles[index] ? speed3 : (speed1 + omega * (d_equ1 - speed1));
  tmp_cells->speeds[index + (2 * totalSize)] = obstacles[index] ? speed4 : (speed2 + omega * (d_equ2 - speed2));
  tmp_cells->speeds[index + (3 * totalSize)] = obstacles[index] ? speed1 : (speed3 + omega * (d_equ3 - speed3));
  tmp_cells->speeds[index + (4 * totalSize)] = obstacles[index] ? speed2 : (speed4 + omega * (d_equ4 - speed4));
  tmp_cells->speeds[index + (5 * totalSize)] = obstacles[index] ? speed7 : (speed5 + omega * (d_equ5 - speed5));
  tmp_cells->speeds[index + (6 * totalSize)] = obstacles[index] ? speed8 : (speed6 + omega * (d_equ6 - speed6));
  tmp_cells->speeds[index + (7 * totalSize)] = obstacles[index] ? speed5 : (speed7 + omega * (d_equ7 - speed7));
  tmp_cells->speeds[index + (8 * totalSize)] = obstacles[index] ? speed6 : (speed8 + omega * (d_equ8 - speed8));

    /* increase counter of inspected cells */

    /* accumulate the norm of x- and y- velocity components */
    l_tot_u[l_x + (l_y * l_size_x)] = (float)sqrt((u_x * u_x) + (u_y * u_y));
    /* increase counter of inspected cells */
    // ++l_tot_cells;
  //}
               

  // barrier(CLK_LOCAL_MEM_FENCE);

  // local float l_av_velocity = l_tot_u / (float) l_tot_cells;

  barrier(CLK_LOCAL_MEM_FENCE);
  
  

  if (l_y == 0 && l_x == 0) {

    int pos_x = ii / l_size_x;
    int pos_y = jj / l_size_y;
    int workgroup = pos_x + (pos_y * work_group_row);

    for (int x = 0; x < l_size_x; x++) {
      for (int y = 0; y < l_size_y; y++) {
        g_tot_u[workgroup + (currentIter * num_workgroup)] += l_tot_u[x + (y * l_size_x)];
      }
    }
    
  }
  
  barrier(CLK_LOCAL_MEM_FENCE);

}