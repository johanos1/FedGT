/*
 * BP decoder for Quantitative Group testing with
 * noiseless tests
 *
 * /
/* C Libraries */

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#define VALID_NEIGH(s) ((s) >= 0)
#define MAXS_H_INTERVAL 0.001
#define MAXS_H_NUMSAMPLES 20000

/* Look up table */
//double jaclog_vals[MAXS_H_NUMSAMPLES];

double jacLog(double d)
{
    return log(1 + exp(-d));
}

/* Needed global variables */
static uint16_t t, n; // t is number of tests, n is number of nodes
static uint8_t max_dv, max_dc;
static int16_t **vn; // neighbors of variable nodes - size (n, max_dv ), -1 for irregular
static int16_t **cn; // neighbors of check nodes - size (t, max_dc), -1 for irregular
static uint8_t *vn_deg, *cn_deg;

/* Needed functions */

static int16_t **malloc2d_si16(size_t nx, size_t ny)
{
    size_t i;
    int16_t **p = malloc(nx * sizeof(int16_t *));
    p[0] = malloc(nx * ny * sizeof(int16_t));
    for (i = 1; i < nx; i++)
        p[i] = p[i - 1] + ny;
    return p;
}

static void free2d_si16(int16_t **p)
{
    free(p[0]);
    free(p);
}

static double **malloc2d_d(size_t nx, size_t ny)
{
    size_t i;
    double **p = malloc(nx * sizeof(double *));
    p[0] = malloc(nx * ny * sizeof(double));
    for (i = 1; i < nx; i++)
        p[i] = p[i - 1] + ny;
    return p;
}

static void free2d_d(double **p)
{
    free(p[0]);
    free(p);
}

static double ***malloc3d_d(size_t nx, size_t ny, size_t nz)
{
    size_t i;
    double ***p = malloc(nx * sizeof(double **));
    p[0] = malloc(nx * ny * sizeof(double *));
    p[0][0] = malloc(nx * ny * nz * sizeof(double));
    for (i = 1; i < nx; i++)
        p[i] = p[i - 1] + ny;
    for (i = 1; i < (nx * ny); i++)
        p[0][i] = p[0][i - 1] + nz;
    return p;
}

static void free3d_d(double ***p)
{
    free(p[0][0]);
    free(p[0]);
    free(p);
}

/* initialize input matrices */

void initializematrix(const int16_t *H, uint64_t nrows, uint64_t ncols, bool variable_nodes)
{
    int16_t **Hmat = malloc2d_si16(nrows, ncols);
    for (size_t i = 0; i < nrows; i++)
    {
        for (size_t j = 0; j < ncols; j++)
        {
            Hmat[i][j] = H[j + i * ncols];
        }
    }
    // printf("U gjeneru1!\n");
    if (variable_nodes)
    {
        for (size_t i = 0; i < nrows; i++)
        {
            for (size_t j = 0; j < ncols; j++)
            {
                vn[i][j] = Hmat[i][j];
            }
        }
    }
    else
    {
        for (size_t i = 0; i < nrows; i++)
        {
            for (size_t j = 0; j < ncols; j++)
            {
                cn[i][j] = Hmat[i][j];
            }
        }
    }
    free2d_si16(Hmat);
}

// compute maxs via LUT
double maxstar(double x, double y)
{

    size_t i = round(abs(x - y) / MAXS_H_INTERVAL);

    if (x == -INFINITY || y == -INFINITY) // i >= MAXS_H_NUMSAMPLES ||
        return (x > y ? x : y);
    else
        return (x > y ? x : y) + jacLog(fabs(x - y)); // jaclog_vals[i];
}

double calculate_sum_product(double *v1, double *v2, int n, int t)
// v1 -> Probabilities for 1, v2 -> Probabilities for 0, n -> length of the vectors, t -> value of tests
{
    if (n <= 0 || t > n)
    {
        perror("Problem with your values of n and t"); // Invalid input, arrays must be of positive length and t should be <= n
        exit(-1);
    }
    double result = -INFINITY;
    size_t i, j;

    if (t == 0)
    {
        double product = 0;
        for (i = 0; i < n; i++)
            product += v2[i];
        return product;
    }
    if (t == n)
    {
        double product = 0;
        for (i = 0; i < n; i++)
            product += v1[i];
        return product;
    }

    for (i = 0; i < (1 << n); ++i)
    { // iterate over all possible subsets of v1
        uint16_t count = 0;
        for (j = 0; j < n; ++j)
        {
            if (i & (1 << j))
            { // check if jth element is chosen in the subset
                ++count;
            }
        }
        if (count == t)
        { // choose subsets with t elements
            double product = 1;
            for (j = 0; j < n; ++j)
            {
                if (i & (1 << j))
                { // check if jth element is chosen in the subset
                    product += v1[j];
                }
                else
                {
                    product += v2[j];
                }
            }
            result = maxstar(result, product);
        }
    }
    return result;
}

double CN_update(double *v1, double *v2, uint8_t degree, int test_value, bool is_0)
{
    if (test_value == 0)
    {
        if (is_0)
            return 0;
        else
            return -INFINITY;
    }
    if ((test_value == degree) && (!is_0))
        return 0;
    if ((test_value == degree) && (is_0))
        return -INFINITY;
    // Change the VN update as in CN
    if (is_0)
    {
        double temp = calculate_sum_product(v1, v2, degree - 1, test_value);
        return temp;
    }
    else
    {
        double temp = calculate_sum_product(v1, v2, degree - 1, test_value - 1);
        return temp;
    }
}

// test_outcome: t size vector where each entry can be 0,1,...,max_dc
void decode(uint8_t *test_outcome, uint8_t *DEC, double *prevalence, uint16_t maxIter)
{
    double ***L_vc = malloc3d_d(max_dv, n, 2);
    double ***L_cv = malloc3d_d(max_dc, t, 2);
    /* Message arrays */
    double **L_cvi = malloc2d_d(2, max_dv);
    double **L_vci = malloc2d_d(2, max_dc);
    /*Temporary arrays for CN operation*/
    double *temp_cn_0 = malloc((max_dc - 1) * sizeof(double));
    double *temp_cn_1 = malloc((max_dc - 1) * sizeof(double));

    /* Initialize the prob vectors */
    for (size_t j = 0; j < max_dv; j++)
    {
        for (size_t i = 0; i < n; i++)
        {
            for (size_t bit = 0; bit < 2; bit++)
                L_vc[j][i][bit] = log(0.5);
        }
    }
    for (size_t j = 0; j < max_dc; j++)
    {
        for (size_t i = 0; i < t; i++)
        {
            for (size_t bit = 0; bit < 2; bit++)
                L_cv[j][i][bit] = 0;
        }
    }

    while (maxIter-- > 0)
    {
        /* VN update */
        for (size_t i = 0; i < n; i++)
        {
            // get messages from neighbors
            for (size_t l = 0; l < vn_deg[i]; l++)
            {
                // neighbor
                int16_t N = vn[i][l];
                if (!VALID_NEIGH(N))
                {
                    exit(-5);
                }
                // get message from this neighbor
                for (size_t j = 0; j < cn_deg[N]; j++)
                {
                    if (!VALID_NEIGH(cn[N][j]))
                    {
                        exit(-5);
                    }
                    if (cn[N][j] == i)
                    {
                        for (size_t bit = 0; bit < 2; bit++)
                            L_cvi[bit][l] = L_cv[j][N][bit]; // cv or vc?
                        break;
                    }
                }
            }
            // update
            double temp_prev = prevalence[i];
            for (size_t l = 0; l < vn_deg[i]; l++)
            {
                if (!VALID_NEIGH(vn[i][l]))
                {
                    exit(-5);
                }
                for (size_t bit = 0; bit < 2; bit++)
                {
                    bool hard_decoded = false;
                    L_vc[l][i][bit] = (bit == 1) ? log(temp_prev) : log(1 - temp_prev); // prevalence
                    for (size_t r = 0; r < vn_deg[i]; r++)
                    {
                        if (r != l)
                            L_vc[l][i][bit] += L_cvi[bit][r]; // VN
                        else
                        {
                            size_t other_bit = (bit == 0) ? 1 : 0;
                            if ((L_cvi[bit][r] == 0) && (L_cvi[other_bit][r] == -INFINITY))
                            {
                                L_vc[l][i][bit] = 0;
                                L_vc[l][i][other_bit] = -INFINITY;
                                hard_decoded = true;
                                break;
                            }
                        }
                    }
                    if (hard_decoded == true)
                        break;
                }
                /* Normalize the entries bit-wise*/
                double temp_sum = maxstar(L_vc[l][i][0], L_vc[l][i][1]);
                if (temp_sum == -INFINITY)
                {
                    for (size_t bit = 0; bit < 2; bit++)
                    {
                        L_vc[l][i][bit] = log(0.5);
                    }
                    // exit(3);
                }
                else
                {
                    for (size_t bit = 0; bit < 2; bit++)
                    {
                        L_vc[l][i][bit] -= temp_sum;
                    }
                }
            }
        }
        /* CN update */
        for (size_t i = 0; i < t; i++)
        {
            // get messages from neighbors
            for (size_t l = 0; l < cn_deg[i]; l++)
            {
                // neighbor
                int16_t N = cn[i][l];
                if (!VALID_NEIGH(N))
                {
                    exit(-5);
                }
                // get message from this neighbor
                for (size_t r = 0; r < vn_deg[N]; r++)
                {
                    if (!VALID_NEIGH(vn[N][r]))
                    {
                        exit(-5);
                    }
                    if (vn[N][r] == i)
                    {
                        for (size_t bit = 0; bit < 2; bit++)
                            L_vci[bit][l] = L_vc[r][N][bit];
                        break;
                    }
                }
            }
            // update
            for (size_t l = 0; l < cn_deg[i]; l++)
            {
                size_t temp_indeks = 0;
                size_t superbit, otherbit;
                bool hard_decoded_cn = false;
                for (size_t r = 0; r < cn_deg[i]; r++)
                {
                    if (r != l)
                    {
                        temp_cn_0[temp_indeks] = L_vci[0][r];
                        temp_cn_1[temp_indeks] = L_vci[1][r];
                        temp_indeks++;
                    }
                    else
                    {
                        if ((L_vci[0][r] == 0) && (L_vci[1][r] == -INFINITY))
                        {
                            superbit = 0;
                            otherbit = 1;
                            hard_decoded_cn = true;
                        }
                        if ((L_vci[1][r] == 0) && (L_vci[0][r] == -INFINITY))
                        {
                            superbit = 1;
                            otherbit = 0;
                            hard_decoded_cn = true;
                        }
                        if (hard_decoded_cn)
                            break;
                    }
                }
                int test_result = test_outcome[i];
                if (!hard_decoded_cn)
                {
                    L_cv[l][i][0] = CN_update(temp_cn_1, temp_cn_0, cn_deg[i], test_result, 1);
                    L_cv[l][i][1] = CN_update(temp_cn_1, temp_cn_0, cn_deg[i], test_result, 0);
                }
                else
                {
                    L_cv[l][i][superbit] = 0;
                    L_cv[l][i][otherbit] = -INFINITY;
                }
                /* Normalize the entries bit-wise*/
                double temp_sum = maxstar(L_cv[l][i][0], L_cv[l][i][1]);
                if (temp_sum == -INFINITY)
                {
                    for (size_t bit = 0; bit < 2; bit++)
                        L_cv[l][i][bit] = log(0.5);
                }
                else
                {
                    for (size_t bit = 0; bit < 2; bit++)
                        L_cv[l][i][bit] -= temp_sum;
                }
            }
        }
        // final APP calculation
        for (size_t i = 0; i < n; i++)
        {
            double temp_1 = log(prevalence[i]);
            double temp_0 = log(1 - prevalence[i]);
            // get messages from neighbors
            for (size_t l = 0; l < vn_deg[i]; l++)
            {
                // neighbor
                int16_t N = vn[i][l];
                if (!VALID_NEIGH(N))
                {
                    exit(-5);
                }
                // add message from this neighbor
                for (size_t r = 0; r < cn_deg[N]; r++)
                {
                    if (!VALID_NEIGH(cn[N][r]))
                    {
                        exit(-5);
                    }
                    if (cn[N][r] == i)
                    {
                        temp_0 += L_cv[r][N][0];
                        temp_1 += L_cv[r][N][1];
                        // break; // neighbour should only be once there!
                    }
                }
            }
            DEC[i] = (temp_1 > temp_0) ? 1 : 0;
        }
        // check, whether the infered vector satisfies the tests.. (maybe for noiseless only)
        bool isCodeword = true;
        for (size_t i = 0; i < t; i++)
        {
            uint8_t parity = 0;
            for (size_t l = 0; l < cn_deg[i]; l++)
                parity += DEC[cn[i][l]]; // wrong
            if (parity != test_outcome[i])
            {
                isCodeword = false;
                break;
            }
        }
        if (isCodeword)
            break; //return
    }
    free3d_d(L_cv);
    free3d_d(L_vc);
    free2d_d(L_cvi);
    free2d_d(L_vci);
    free(temp_cn_0);
    free(temp_cn_1);
}
// interface
void BP_decoder(const uint16_t n_input, const uint16_t t_input, const uint8_t dvmax, const uint8_t dcmax,
                const int16_t *vn_input, const int16_t *cn_input, const uint8_t *test_outcome_input,
                const double *prevalence_input, const uint16_t max_Iter_input, const uint8_t *cn_deg_input, const uint8_t *vn_deg_input, uint8_t *DEC)
{
    n = n_input;
    t = t_input;
    max_dv = dvmax;
    max_dc = dcmax;
    vn = malloc2d_si16(n, max_dv);
    cn = malloc2d_si16(t, max_dc);
    initializematrix(vn_input, n, max_dv, 1);
    initializematrix(cn_input, t, max_dc, 0);
    cn_deg = malloc(t * sizeof(uint8_t));
    for (size_t i = 0; i < t; i++)
        cn_deg[i] = cn_deg_input[i]; // works for regular only
    vn_deg = malloc(n * sizeof(uint8_t));
    for (size_t i = 0; i < n; i++)
        vn_deg[i] = vn_deg_input[i]; // works for regular only
    uint8_t *test_outcome = malloc(t * sizeof(uint8_t));
    for (size_t i = 0; i < t; i++)
        test_outcome[i] = test_outcome_input[i];
    double *prevalence = malloc(n * sizeof(double));
    for (size_t i = 0; i < n; i++)
        prevalence[i] = prevalence_input[i];
    uint16_t maxIter = max_Iter_input;
    /*for (size_t i = 0; i < MAXS_H_NUMSAMPLES; i++)
    {
        jaclog_vals[i] = jacLog(i * MAXS_H_INTERVAL);
    }*/
    decode(test_outcome, DEC, prevalence, maxIter);
    free2d_si16(vn);
    free2d_si16(cn);
    free(test_outcome);
    free(prevalence);
    free(cn_deg);
    free(vn_deg);
}
