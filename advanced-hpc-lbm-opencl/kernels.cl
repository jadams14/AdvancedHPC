#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9


kernel void accelerate_flow(global float* restrict cells,
                            global int* restrict obstacles,
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
  const int totalSize = nx * ny;
  int index = ii + jj*nx;
    if (!obstacles[index]
        && (cells[index + (3 * totalSize)] - w1) > 0.f
        && (cells[index + (6 * totalSize)] - w2) > 0.f
        && (cells[index + (7 * totalSize)] - w2) > 0.f)
    {
      /* increase 'east-side' densities */
      cells[index + (1 * totalSize)] += w1;
      cells[index + (5 * totalSize)] += w2;
      cells[index + (8 * totalSize)] += w2;
      /* decrease 'west-side' densities */
      cells[index + (3 * totalSize)] -= w1;
      cells[index + (6 * totalSize)] -= w2;
      cells[index + (7 * totalSize)] -= w2;
    }
  
}

kernel void collision(global float * restrict cells, 
                      global float * restrict tmp_cells, 
                      global int * restrict obstacles, 
                      global float * restrict g_tot_u,
                      local float * restrict l_tot_u,
                      int nx, int ny,
                      float omega, 
                      int currentIter,
                      float density, float accel) {

  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;   /* weighting factor */
  const float w1 = 1.f / 9.f;   /* weighting factor */
  const float w2 = 1.f / 36.f;  /* weighting factor */
  const float w1_a_f = density * accel / 9.0;
  const float w2_a_f = density * accel / 36.0;

  /* get column and row indices */
  const int ii = get_global_id(0);
  const int jj = get_global_id(1);
  const int g_size_ii = get_global_size(0);
  const int g_size_jj = get_global_size(1);
  const int l_x = get_local_id(0);
  const int l_y = get_local_id(1);
  const int x = get_group_id(0);
  const int y = get_group_id(1);
  const int l_size_x = get_local_size(0);
  const int l_size_y = get_local_size(1);
  const int workgroup = x + (y * get_num_groups(0));
  const int local_index = (l_x + (l_y * l_size_x));
  const int cell_index = ii + jj * nx;
  const int size_workgroup = l_size_x * l_size_y;
        
  const int y_n = (jj+1) - ny *((jj + 1) == ny);
  const int x_e = (ii+1) - nx * ((ii + 1) == nx);
  const int y_s = ((jj == 0) * ny) + jj - 1;
  const int x_w = ((ii == 0) * nx) + ii - 1;
  //Propogate
  const int totalSize = nx * ny;
  const int mask1 = ((!obstacles[x_w + jj*nx])
        && ((cells[x_w + jj*nx + (3 * totalSize)] - w1_a_f) > 0.f)
        && ((cells[x_w + jj*nx + (6 * totalSize)] - w2_a_f) > 0.f)
        && ((cells[x_w + jj*nx + (7 * totalSize)] - w2_a_f) > 0.f)
        && (jj == ny - 2));
  const int mask2 = ((!obstacles[x_e + jj*nx])
        && ((cells[x_e + jj*nx + (3 * totalSize)] - w1_a_f) > 0.f)
        && ((cells[x_e + jj*nx + (6 * totalSize)] - w2_a_f) > 0.f)
        && ((cells[x_e + jj*nx + (7 * totalSize)] - w2_a_f) > 0.f)
        && (jj == ny - 2));
  const int mask3 = ((!obstacles[x_w + y_s*nx])
        && ((cells[x_w + y_s*nx + (3 * totalSize)] - w1_a_f) > 0.f)
        && ((cells[x_w + y_s*nx + (6 * totalSize)] - w2_a_f) > 0.f)
        && ((cells[x_w + y_s*nx + (7 * totalSize)] - w2_a_f) > 0.f)
        && (y_s == ny - 2));
  const int mask4 = ((!obstacles[x_e + y_s*nx])
        && ((cells[x_e + y_s*nx + (3 * totalSize)] - w1_a_f) > 0.f)
        && ((cells[x_e + y_s*nx + (6 * totalSize)] - w2_a_f) > 0.f)
        && ((cells[x_e + y_s*nx + (7 * totalSize)] - w2_a_f) > 0.f)
        && (y_s == ny - 2));
  const int mask5 = ((!obstacles[x_e + y_n*nx])
        && ((cells[x_e + y_n*nx + (3 * totalSize)] - w1_a_f) > 0.f)
        && ((cells[x_e + y_n*nx + (6 * totalSize)] - w2_a_f) > 0.f)
        && ((cells[x_e + y_n*nx + (7 * totalSize)] - w2_a_f) > 0.f)
        && (y_n == ny - 2));
  const int mask6 = ((!obstacles[x_w + y_n*nx])
        && ((cells[x_w + y_n*nx + (3 * totalSize)] - w1_a_f) > 0.f)
        && ((cells[x_w + y_n*nx + (6 * totalSize)] - w2_a_f) > 0.f)
        && ((cells[x_w + y_n*nx + (7 * totalSize)] - w2_a_f) > 0.f)
        && (y_n == ny - 2));

  const float speed = cells[cell_index];
  const float speed1 = cells[x_w + jj*nx + (1 * totalSize)] + (mask1 * w1_a_f);
  const float speed2 = cells[ii + y_s*nx + (2 * totalSize)];
  const float speed3 = cells[x_e + jj*nx + (3 * totalSize)] - (mask2 * w1_a_f);
  const float speed4 = cells[ii + y_n*nx + (4 * totalSize)];
  const float speed5 = cells[x_w + y_s*nx + (5 * totalSize)] + (mask3 * w2_a_f);
  const float speed6 = cells[x_e + y_s*nx + (6 * totalSize)] - (mask4 * w2_a_f);
  const float speed7 = cells[x_e + y_n*nx + (7 * totalSize)] - (mask5 * w2_a_f);
  const float speed8 = cells[x_w + y_n*nx + (8 * totalSize)] + (mask6 * w2_a_f);
  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  
    /* compute local density total */
    const float local_density = speed + speed1 + speed2 + speed3 + speed4
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
    const float u_sq = u_x * u_x + u_y * u_y;
    /* directional velocity components */
    
    const float u1 = u_x;        /* east */
    const float u2 = u_y;        /* north */
    const float u3 = -u_x;       /* west */
    const float u4 = -u_y;       /* south */
    const float u5 = u_x + u_y;  /* north-east */
    const float u6 = -u_x + u_y; /* north-west */
    const float u7 = -u_x - u_y; /* south-west */
    const float u8 = u_x - u_y;  /* south-east */

    const float localDensityW1 = w1 * local_density;
    const float localDensityW2 = w2 * local_density;
    const float timesedC_SQ    = (2.f * c_sq * c_sq);
    const float minusU_SQ      = u_sq / (2.f * c_sq);

    const float d_equ = (w0 * local_density
                          * (1.f - minusU_SQ));
    const float d_equ1 = (localDensityW1 * (1.f + u1 / c_sq
                                     + (u1 * u1) / (timesedC_SQ)
                                     - minusU_SQ));
    const float d_equ2 = (localDensityW1 * (1.f + u2 / c_sq
                                     + (u2 * u2) / (timesedC_SQ)
                                     - minusU_SQ));
    const float d_equ3 = (localDensityW1 * (1.f + u3 / c_sq
                                     + (u3 * u3) / (timesedC_SQ)
                                     - minusU_SQ));
    const float d_equ4 = (localDensityW1 * (1.f + u4 / c_sq
                                     + (u4 * u4) / (timesedC_SQ)
                                     - minusU_SQ));
    const float d_equ5 = (localDensityW2 * (1.f + u5 / c_sq
                                     + (u5 * u5) / (timesedC_SQ)
                                     - minusU_SQ));
    const float d_equ6 = (localDensityW2 * (1.f + u6 / c_sq
                                     + (u6 * u6) / (timesedC_SQ)
                                     - minusU_SQ));
    const float d_equ7 = (localDensityW2 * (1.f + u7 / c_sq
                                   + (u7 * u7) / (timesedC_SQ)
                                     - minusU_SQ));
    const float d_equ8 = (localDensityW2 * (1.f + u8 / c_sq
                                     + (u8 * u8) / (timesedC_SQ)
                                     - minusU_SQ));

    // /* relaxation step */
    // tmp_cells[cell_index]  = (obstacles[cell_index]) ? speed : (speed + omega * (d_equ - speed));
    // tmp_cells[cell_index  + (1 * totalSize)] = (obstacles[cell_index]) ? speed3 : (speed1 + omega * (d_equ1 - speed1));
    // tmp_cells[cell_index  + (2 * totalSize)] = (obstacles[cell_index]) ? speed4 : (speed2 + omega * (d_equ2 - speed2));
    // tmp_cells[cell_index  + (3 * totalSize)] = (obstacles[cell_index]) ? speed1 : (speed3 + omega * (d_equ3 - speed3));
    // tmp_cells[cell_index  + (4 * totalSize)] = (obstacles[cell_index]) ? speed2 : (speed4 + omega * (d_equ4 - speed4));
    // tmp_cells[cell_index  + (5 * totalSize)] = (obstacles[cell_index]) ? speed7 : (speed5 + omega * (d_equ5 - speed5));
    // tmp_cells[cell_index  + (6 * totalSize)] = (obstacles[cell_index]) ? speed8 : (speed6 + omega * (d_equ6 - speed6));
    // tmp_cells[cell_index  + (7 * totalSize)] = (obstacles[cell_index]) ? speed5 : (speed7 + omega * (d_equ7 - speed7));
    // tmp_cells[cell_index  + (8 * totalSize)] = (obstacles[cell_index]) ? speed6 : (speed8 + omega * (d_equ8 - speed8));
    tmp_cells[cell_index]  = ((obstacles[cell_index]) * speed) + ((!obstacles[cell_index]) * (speed + omega * (d_equ - speed)));
    tmp_cells[cell_index  + (1 * totalSize)] = ((obstacles[cell_index]) * speed3) + ((!obstacles[cell_index]) * (speed1 + omega * (d_equ1 - speed1)));
    tmp_cells[cell_index  + (2 * totalSize)] = ((obstacles[cell_index]) * speed4) + ((!obstacles[cell_index]) * (speed2 + omega * (d_equ2 - speed2)));
    tmp_cells[cell_index  + (3 * totalSize)] = ((obstacles[cell_index]) * speed1) + ((!obstacles[cell_index]) * (speed3 + omega * (d_equ3 - speed3)));
    tmp_cells[cell_index  + (4 * totalSize)] = ((obstacles[cell_index]) * speed2) + ((!obstacles[cell_index]) * (speed4 + omega * (d_equ4 - speed4)));
    tmp_cells[cell_index  + (5 * totalSize)] = ((obstacles[cell_index]) * speed7) + ((!obstacles[cell_index]) * (speed5 + omega * (d_equ5 - speed5)));
    tmp_cells[cell_index  + (6 * totalSize)] = ((obstacles[cell_index]) * speed8) + ((!obstacles[cell_index]) * (speed6 + omega * (d_equ6 - speed6)));
    tmp_cells[cell_index  + (7 * totalSize)] = ((obstacles[cell_index]) * speed5) + ((!obstacles[cell_index]) * (speed7 + omega * (d_equ7 - speed7)));
    tmp_cells[cell_index  + (8 * totalSize)] = ((obstacles[cell_index]) * speed6) + ((!obstacles[cell_index]) * (speed8 + omega * (d_equ8 - speed8)));

  // /* accumulate the norm of x- and y- velocity components */
    l_tot_u[local_index] = (obstacles[cell_index]) ? 0 : (float)sqrt((u_x * u_x) + (u_y * u_y));
    // l_tot_u[local_index] = (obstacles[cell_index]) ? 0 : (float)sqrt((u_x * u_x) + (u_y * u_y));
  /* increase counter of inspected cells */
  // ++l_tot_cells;

  for (int i = 1; i < size_workgroup; i *= 2) {
    int index = local_index * i;
    if (index < size_workgroup) {
      l_tot_u[index] += l_tot_u[index + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  // for  (int i = size_workgroup/2; i > 0; i >>= 1) {
  //   if (local_index < (i/2)) {
      // float first = l_tot_u[local_index + i];
      // float second = l_tot_u[local_index];
      // l_tot_u[local_index] = (second < first) ? second : first;
  //     l_tot_u[local_index] += l_tot_u[local_index + (i / 2)];
  //   }
  //   barrier(CLK_LOCAL_MEM_FENCE);
  // }
  if (local_index == 0) {
    int num_workgroup = (get_num_groups(0) * get_num_groups(1));
    int workgroup = x + (y * get_num_groups(0));
    g_tot_u[workgroup + (currentIter * num_workgroup)] = l_tot_u[0];
    // if (g_tot_u[workgroup + (currentIter * num_workgroup)] == 0) {
      // printf("0 here: %d", workgroup + (currentIter * num_workgroup));
    // }
  }
}
                    