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
  if (!obstacles[ii + jj* nx]
      && (cells[ii + jj* nx].speeds[3] - w1) > 0.f
      && (cells[ii + jj* nx].speeds[6] - w2) > 0.f
      && (cells[ii + jj* nx].speeds[7] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cells[ii + jj* nx].speeds[1] += w1;
    cells[ii + jj* nx].speeds[5] += w2;
    cells[ii + jj* nx].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii + jj* nx].speeds[3] -= w1;
    cells[ii + jj* nx].speeds[6] -= w2;
    cells[ii + jj* nx].speeds[7] -= w2;
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
                      float omega, 
                      int currentIter, 
                      int divide) {

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */

  

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  
  /* get column and row indices */
  int ii = get_global_id(0);
  int jj = get_global_id(1);
  int g_size_ii = get_global_size(0);
  int g_size_jj = get_global_size(1);
  int l_x = get_local_id(0);
  int l_y = get_local_id(1);
  int x = get_group_id(0);
  int y = get_group_id(1);

  int l_size_x = get_local_size(0);
  int l_size_y = get_local_size(1);

  int workgroup = x + (y * divide);


  // //Used to store the local workgroups tot_u
  // local int l_tot_cells;
  // for (int i = 0; i < l_size_x; i++) {
  //   for (int j = 0; j < l_size_y; j++) {
  //     l_tot_u[i + (l_size_x * j)] = 0;
  //   }
  // }
  

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
  /* don't consider occupied cells */
  else if (!obstacles[ii + jj * nx])
  {
    /* compute local density total */
    float local_density = 0.f;
    for (int kk = 0; kk < NSPEEDS; kk++)
    {
      local_density += tmp_cells[ii + jj * nx].speeds[kk];
    }

    /* compute x velocity component */
    float u_x = (tmp_cells[ii + jj * nx].speeds[1] + tmp_cells[ii + jj * nx].speeds[5] + tmp_cells[ii + jj * nx].speeds[8] - (tmp_cells[ii + jj * nx].speeds[3] + tmp_cells[ii + jj * nx].speeds[6] + tmp_cells[ii + jj * nx].speeds[7])) / local_density;
    /* compute y velocity component */
    float u_y = (tmp_cells[ii + jj * nx].speeds[2] + tmp_cells[ii + jj * nx].speeds[5] + tmp_cells[ii + jj * nx].speeds[6] - (tmp_cells[ii + jj * nx].speeds[4] + tmp_cells[ii + jj * nx].speeds[7] + tmp_cells[ii + jj * nx].speeds[8])) / local_density;

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

    /* equilibrium densities */
    float d_equ[NSPEEDS];
    /* zero velocity density: weight w0 */
    d_equ[0] = w0 * local_density * (1.f - u_sq / (2.f * c_sq));
    /* axis speeds: weight w1 */
    d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq + (u[1] * u[1]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
    d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq + (u[2] * u[2]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
    d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq + (u[3] * u[3]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
    d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq + (u[4] * u[4]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
    /* diagonal speeds: weight w2 */
    d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq + (u[5] * u[5]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
    d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq + (u[6] * u[6]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
    d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq + (u[7] * u[7]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
    d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq + (u[8] * u[8]) / (2.f * c_sq * c_sq) - u_sq / (2.f * c_sq));
    
    /* relaxation step */
    for (int kk = 0; kk < NSPEEDS; kk++) {
      cells[ii + jj * nx].speeds[kk] = tmp_cells[ii + jj * nx].speeds[kk] + omega * (d_equ[kk] - tmp_cells[ii + jj *nx].speeds[kk]);
    }

    /* accumulate the norm of x- and y- velocity components */
    l_tot_u[l_x + (l_y * l_size_x)] = (float)sqrt((u_x * u_x) + (u_y * u_y));
    // printf("%f %f %f %f %f %f %f %f %f\n", cells[ii + jj * nx].speeds[0], cells[ii + jj * nx].speeds[1],
    //                                      cells[ii + jj * nx].speeds[2], cells[ii + jj * nx].speeds[3],
    //                                      cells[ii + jj * nx].speeds[4], cells[ii + jj * nx].speeds[5],
    //                                      cells[ii + jj * nx].speeds[6], cells[ii + jj * nx].speeds[7],
    //                                      cells[ii + jj * nx].speeds[8]);
    /* increase counter of inspected cells */
    // ++l_tot_cells;

  }
               

  // barrier(CLK_LOCAL_MEM_FENCE);

  // local float l_av_velocity = l_tot_u / (float) l_tot_cells;

  barrier(CLK_LOCAL_MEM_FENCE);
  
  

  if (l_y == 0 && l_x == 0) {
    int num_workgroup = get_num_groups(0) * get_num_groups(1);

    for (int xx = 0; xx < l_size_x; xx++) {
      for (int yy = 0; yy < l_size_y; yy++) {
        g_tot_u[workgroup + (currentIter * num_workgroup)] += l_tot_u[xx + (yy * l_size_x)];
      }
      
    }
    // printf("%f\n", g_tot_u[workgroup + (currentIter * num_workgroup)]);
  }
  

}