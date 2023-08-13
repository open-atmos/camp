module camp_debug_2

  use mpi
  use camp_constants,                  only : i_kind, dp
  implicit none

  integer :: rank,ierr,n_ranks,comm

contains

  subroutine init_export_f_state(comm_0)
    integer, intent(in), optional :: comm_0
    comm=MPI_COMM_WORLD
    if(present(comm_0)) then
      comm=comm_0
    end if
    print*,comm
    call mpi_comm_rank(comm, rank, ierr)
    if(rank==0) then
      open(50, file="out/state.csv", status="replace", action="write")
      close(50)
    end if
  end subroutine

  subroutine export_f_state(x, len, len2)
    real(kind=dp), dimension(:), intent(in) :: x
    integer, intent(in) :: len, len2
    integer :: k,j,i
    call mpi_comm_rank(comm, rank, ierr)
    do k=0,len2-1!camp_core%n_cells
      if(rank==0) then
        open(50, file="out/state.csv", status="old", position="append", action="write")
        do i=1, len!camp_core%size_state_per_cell
          write(50, "(ES23.15)") x(i+k*len) !camp_state%state_var(i)
        end do
        close(50)
      end if
    end do
  end subroutine

  subroutine old_export_f_state(x, len, len2) ! fails in MONARCH
    real(kind=dp), dimension(:), intent(in) :: x
    integer, intent(in) :: len, len2
    integer :: k,j,i
    print*,comm
    call mpi_comm_rank(comm, rank, ierr)
    call mpi_comm_size(comm, n_ranks, ierr)
    do k=0,len2-1!camp_core%n_cells
      do j=0, n_ranks-1
        if(rank==j) then
          open(50, file="out/state.csv", status="old", position="append", action="write")
          do i=1, len!camp_core%size_state_per_cell
            write(50, "(ES23.15)") x(i+k*len) !camp_state%state_var(i)
          end do
          close(50)
        end if
        call mpi_barrier(comm, ierr)
      end do
    end do
  end subroutine

end module