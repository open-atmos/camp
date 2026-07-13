module boxmodel_sobol_numbers
#ifdef CAMP_USE_MPI
  use mpi, only: MPI_COMM_WORLD
#endif

  use boxmodel_montecarlo, only : RANDOM, SOBOL

  use camp_util, only: i_kind, die_msg, to_string, dp
  use camp_mpi, only: camp_mpi_rank, camp_mpi_transfer_real, camp_mpi_bcast_integer
  use mod_sobseq, only: MAX_SOBOL_NUMBERS, MAX_DIM, sobol_state

  use boxmodel_log
  implicit none
  private
  public :: next_sobol_number, generate_sobol_numbers, broadcast_sobol_numbers


  integer(kind=i_kind) :: sobol_dim, sobol_samples, sobol_index
  real(kind=dp), dimension(:), allocatable :: sobol_numbers
  real(kind=dp), dimension(:), allocatable :: local_sobol_numbers
contains

  subroutine generate_sobol_numbers(dim, ncores , sobol_offset, opt_debug)
    integer(kind=i_kind), intent(in) :: dim !< the number of dimensions to generate, one for each montecarlo object in the model
    integer(kind=i_kind), intent(in) :: ncores !< the number of samples to generate, one for each process
    integer(kind=i_kind), intent(in) :: sobol_offset !< offset to start the sobol sequences
    logical, intent(in), optional :: opt_debug !< set to `.true.` to print the generated sequence
    logical :: debug
    integer(kind=dp) :: i, j, debug_file
    real(kind=dp) :: dummy

    type(sobol_state), dimension(dim) :: rng

    if (present(opt_debug)) then
      debug = opt_debug
    else
      debug = .FALSE.
    endif

#ifndef CAMP_USE_MPI
    ! this module can only be use in an MPI context
    call die_msg(355475390, " error, trying to call generate_sobol_numbers outside of an MPI context")
#endif
    sobol_dim = dim
    sobol_samples = ncores

    if (camp_mpi_rank() > 0) then
      call die_msg(355475391, " error, trying to call generate_sobol_numbers from process rank="//&
        to_string(camp_mpi_rank()) // ", can only be called by rank=0")
    endif

    if ((sobol_samples + sobol_offset) > MAX_SOBOL_NUMBERS) then
      call die_msg(355475392, " error in generate_sobol_numbers, asking for too many sobol numbers"// &
        to_string(sobol_samples + sobol_offset)//">"//to_string(MAX_SOBOL_NUMBERS))
    endif

    if (sobol_dim > MAX_DIM) then
      call die_msg(355475393, " error in generate_sobol_numbers, using too many dimensions"// &
        to_string(sobol_dim)//">"//to_string(MAX_DIM)//" : look into mod_sobseq.f90 and follow the instruction to add more direction numbers")
    endif

    ! finally initialize the sobol module and generate the numbers that are needed
    do i = 1, sobol_dim
      call rng(i)%initialize(i)
    enddo

    ! offset if needed
    if (sobol_offset > 0) then
      do i = 1, sobol_dim 
        dummy = rng(i)%skip_ahead(sobol_offset)
      enddo 
    endif

    allocate(sobol_numbers(sobol_dim * sobol_samples ))
    do j = 1, sobol_samples
      do i = 1, sobol_dim
        sobol_numbers((j-1)*sobol_dim + i) = rng(i)%next()
      enddo
    enddo

    if (debug) then
      open(newunit = debug_file, file = "debug_sobol_sequence")
      do i = 1, sobol_samples
        write(debug_file, *) sobol_numbers(((i - 1)*sobol_dim + 1):(i*sobol_dim))
      enddo
      close(debug_file)
    endif

  end subroutine generate_sobol_numbers

  subroutine broadcast_sobol_numbers(mpi_comm)
    integer(kind=i_kind), intent(in), optional :: mpi_comm

    integer(kind=i_kind) :: local_comm
    integer(kind=i_kind) :: i, j

    call thread_log%debug("in broadcast_sobol_numbers")
    if (present(mpi_comm)) then
      local_comm = mpi_comm
    else
      local_comm = MPI_COMM_WORLD
    endif

    call camp_mpi_bcast_integer(sobol_dim, local_comm)
    call camp_mpi_bcast_integer(sobol_samples, local_comm)
      
    call thread_log%debug("allocated space for "//to_string(sobol_dim)//" sobol numbers")
    allocate(local_sobol_numbers(sobol_dim))
    local_sobol_numbers = 0.0
    sobol_index = 1

    do i = 0, sobol_samples - 1
      do j = 1, sobol_dim
        call camp_mpi_transfer_real( &
          sobol_numbers(i*sobol_dim + j), &
          local_sobol_numbers(j), &
          0, &
          i )
      enddo
    enddo

    if (camp_mpi_rank(local_comm) /= 0) then
      do i = 1, sobol_dim
        call thread_log%debug("received sobol number "//to_string(local_sobol_numbers(i)))
      enddo
    endif

    ! TODO: add a to_string implementation for real(:), allocatable
    ! call thread_log%debug("process "//to_string(camp_mpi_rank())// &
    !   " : received sobol numbers:"//to_string(local_sobol_numbers))

  end subroutine broadcast_sobol_numbers

  function next_sobol_number()
    real(kind=dp) :: next_sobol_number
    if (sobol_index > sobol_dim) then
      call die_msg(138530880, " error in next_sobol_number, asking for too many sobol numbers")
    endif
    next_sobol_number = local_sobol_numbers(sobol_index)
    sobol_index = sobol_index + 1
  end function next_sobol_number

end module boxmodel_sobol_numbers
