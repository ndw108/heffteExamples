#include "heffte.h"

/*!
 * \brief HeFFTe example 5, computing the Cosine Transform DCT using an arbitrary number of MPI ranks.
 *
 * Performing the discrete Cosine Transform (DCT) on three dimensional data in a box of 64 by 64 by 64
 * split across an arbitrary number of MPI ranks.
 */
void compute_dct(MPI_Comm comm){

    int me; // this process rank within the comm
    MPI_Comm_rank(comm, &me);

    int num_ranks; // total number of ranks in the comm
    MPI_Comm_size(comm, &num_ranks);

    // Using input configuration with pencil data format in X direction
    // and output configuration with pencil data in the Z direction.
    // This format uses only two internal reshape operation.
    std::array<int, 2> proc_grid = heffte::make_procgrid(num_ranks);
    std::array<int, 3> input_grid = {1, proc_grid[0], proc_grid[1]};
    std::array<int, 3> output_grid = {proc_grid[0], proc_grid[1], 1};

    //Define global size 
    int ni=256;
    int nj=96;
    int nk=128;

    // Describe all the indexes across all ranks
    heffte::box3d<> const world = {{0, 0, 0}, {(ni-1), (nj-1), (nk-1)}};

    // Split the world box into a 2D grid of boxes
    std::vector<heffte::box3d<>> inboxes  = heffte::split_world(world, input_grid);
    std::vector<heffte::box3d<>> outboxes = heffte::split_world(world, output_grid);

    // Select the backend to use, prefer FFTW and fallback to the stock backend
    // The real-to-real transforms have _cos and _sin appended
    #ifdef Heffte_ENABLE_FFTW
    using backend_tag = heffte::backend::fftw_cos1;
    #else
    using backend_tag = heffte::backend::stock_cos1;
    #endif

    // define the heffte class and the input and output geometry
    // note that rtransform is just an alias to fft3d
    heffte::rtransform<heffte::backend::fftw_cos1> tcos(inboxes[me], outboxes[me], comm);
    heffte::fft3d<heffte::backend::fftw> fft(inboxes[me], outboxes[me], comm);


    //For the operation fft[i],fft[j],dct[k]
    tcos.nullify_executor(0);
    tcos.nullify_executor(1);

    fft.nullify_executor(2);


    double dx=0.1; 
    int world_plane = nj * ni;
    int world_stride = ni;
    std::vector<double> world_input(ni * nj * nk);
    for( int i=0; i<ni; i++ )
        for( int j=0; j<nj; j++ )
            for( int k=0; k<nk; k++ )
                world_input[k* world_plane + j * world_stride + i] = std::sin(dx*i) + std::cos(dx*(i-j+k));


    std::vector<double> input(tcos.size_inbox());

    // set the strides for the triple indexes
    int local_plane = inboxes[me].size[0] * inboxes[me].size[1];
    int local_stride = inboxes[me].size[0];
    // note the order of the loops corresponding to the default order (0, 1, 2)
    // order (0, 1, 2) means that the data in dimension 0 is contiguous
    for(int i=inboxes[me].low[2]; i <= inboxes[me].high[2]; i++)
        for(int j=inboxes[me].low[1]; j <= inboxes[me].high[1]; j++)
            for(int k=inboxes[me].low[0]; k <= inboxes[me].high[0]; k++)
                input[(i - inboxes[me].low[2]) * local_plane
                      + (j - inboxes[me].low[1]) * local_stride + k - inboxes[me].low[0]]
                    = world_input[i * world_plane + j * world_stride + k];


    std::cout<<me<<" ";
    for(size_t i=0; i<10; i++)
        std::cout<<input[i]<<" ";
    std::cout<<std::endl;



    // vectors with the correct sizes to store the input and output data
    // taking the size of the input and output boxes
    std::vector<double> output(tcos.size_outbox());
    std::vector<std::complex<double>> output_complex(fft.size_outbox());

    // the workspace vector is of a real type too
    std::vector<double> workspace(tcos.size_workspace());


    tcos.forward(input.data(), output.data());
    fft.forward(output.data(), output_complex.data());


    // compute the inverse or backward transform
    std::vector<double> inverse_fft(fft.size_outbox());
    std::vector<double> inverse(tcos.size_outbox());

    fft.backward(output_complex.data(), inverse_fft.data());
    tcos.backward(inverse_fft.data(), inverse_fft.data(), workspace.data());

    for(size_t i=0; i<inverse_fft.size(); i++) {
        inverse_fft[i] /= 2*ni*nj*(nk-1);
    }

    std::cout<<me<<" ";
    for(size_t i=0; i<10; i++)
        std::cout<<inverse_fft[i]<<" ";
    std::cout<<std::endl;


    double err = 0.0;
    for(size_t i=0; i<inverse.size(); i++)
        err = std::max(err, std::abs(inverse_fft[i] - input[i]));

    // print the error for each MPI rank
    std::cout << std::scientific;
    for(int i=0; i<num_ranks; i++){
        if (me == i) std::cout << "rank " << i << " error: " << err << std::endl;
        MPI_Barrier(comm);
    }
}

int main(int argc, char** argv){

    MPI_Init(&argc, &argv);

    compute_dct(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
