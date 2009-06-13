! Copyright (C) 2005-2009 Nicole Riemer and Matthew West
! Licensed under the GNU General Public License version 2 or (at your
! option) any later version. See the file COPYING for the.

!> \file
!> The pmc_aero_binned module.

!> The aero_binned_t structure and associated subroutines.
module pmc_aero_binned

  use pmc_bin_grid
  use pmc_aero_particle
  use pmc_spec_read
  use pmc_util
  use pmc_bin_grid
  use pmc_aero_dist
  use pmc_mpi
  use pmc_aero_data
#ifdef PMC_USE_MPI
  use mpi
#endif

  !> Aerosol number and volume distributions stored per bin.
  !!
  !! These quantities are densities both in volume (per m^3) and in
  !! radius (per dlnr). The total concentration per volume is computed as
  !! sum(aero_binned\%num_conc * bin_grid\%dlnr).
  !!
  !! An aero_binned_t is similar to an aero_dist_t in that they both
  !! store binned aerosol distributions. The difference is that an
  !! aero_dist_t has the same composition in every bin, whereas an
  !! aero_binned_t can have aerosol composition that varies per bin.
  type aero_binned_t
     !> Number concentration per bin (#/m^3/dlnr).
     !! Array length is typically \c bin_grid\%n_bin.
     real*8, pointer :: num_conc(:)
     !> Volume concentration per bin and per species (m^3/m^3/dlnr).
     !! Array size is typically \c bin_grid\%n_bin x \c aero_data\%n_spec.
     real*8, pointer :: vol_conc(:,:)
  end type aero_binned_t

contains
  
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Allocate an aero_binned_t.
  subroutine aero_binned_allocate(aero_binned)

    !> Structure to be allocated.
    type(aero_binned_t), intent(out) :: aero_binned

    allocate(aero_binned%num_conc(0))
    allocate(aero_binned%vol_conc(0, 0))

  end subroutine aero_binned_allocate

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Allocate an aero_binned_t of the given size.
  subroutine aero_binned_allocate_size(aero_binned, n_bin, n_spec)

    !> Structure to be allocated.
    type(aero_binned_t), intent(out) :: aero_binned
    !> Number of aerosol bins to allocate (typically \c bin_grid%%n_bin).
    integer, intent(in) :: n_bin
    !> Number of aerosol species to allocate (typically \c aero_data%%n_spec).
    integer, intent(in) :: n_spec

    allocate(aero_binned%num_conc(n_bin))
    allocate(aero_binned%vol_conc(n_bin, n_spec))
    call aero_binned_zero(aero_binned)

  end subroutine aero_binned_allocate_size

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Free internal memory in an aero_binned_t structure.
  subroutine aero_binned_deallocate(aero_binned)

    !> Structure to free.
    type(aero_binned_t), intent(inout) :: aero_binned

    deallocate(aero_binned%num_conc)
    deallocate(aero_binned%vol_conc)

  end subroutine aero_binned_deallocate

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Set all internal data in an aero_binned_t structure to zero.
  subroutine aero_binned_zero(aero_binned)

    !> Structure to zero.
    type(aero_binned_t), intent(inout) :: aero_binned

    aero_binned%num_conc = 0d0
    aero_binned%vol_conc = 0d0

  end subroutine aero_binned_zero

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Update aero_binned_t structure for the addition of the given
  !> particle whose bin is also given.
  !!
  !! If the bin of the particle is not known the more expensive
  !! aero_binned_add_particle() can be used.
  subroutine aero_binned_add_particle_in_bin(aero_binned, bin_grid, &
       bin, comp_vol, aero_particle)

    !> Structure to update with the new particle.
    type(aero_binned_t), intent(inout) :: aero_binned
    !> Bin grid.
    type(bin_grid_t), intent(in) :: bin_grid
    !> Bin number that new particle is in (must be correct).
    integer, intent(in) :: bin
    !> Computational volume (m^3).
    real*8, intent(in) :: comp_vol
    !> Particle to add.
    type(aero_particle_t), intent(in) :: aero_particle

    aero_binned%num_conc(bin) = aero_binned%num_conc(bin) &
         + 1d0 / comp_vol / bin_grid%dlnr
    aero_binned%vol_conc(bin,:) = aero_binned%vol_conc(bin,:) &
         + aero_particle%vol / comp_vol / bin_grid%dlnr

  end subroutine aero_binned_add_particle_in_bin

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Update aero_binned_t structure for the addition of the given
  !> particle.
  !!
  !! If the correct bin for the particle is already known then it is
  !! cheaper to call aero_binned_add_particle_in_bin().
  subroutine aero_binned_add_particle(aero_binned, bin_grid, &
       comp_vol, aero_particle)

    !> Structure to update with the new particle.
    type(aero_binned_t), intent(inout) :: aero_binned
    !> Bin grid.
    type(bin_grid_t), intent(in) :: bin_grid
    !> Computational volume (m^3).
    real*8, intent(in) :: comp_vol
    !> Particle to add.
    type(aero_particle_t), intent(in) :: aero_particle

    call aero_binned_add_particle_in_bin(aero_binned, bin_grid, &
         aero_particle_in_bin(aero_particle, bin_grid), comp_vol, &
         aero_particle)

  end subroutine aero_binned_add_particle

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Update aero_binned_t structure for the removal of the given
  !> particle whose bin is also given.
  !!
  !! If the bin of the particle is not known the more expensive
  !! aero_binned_remove_particle() can be used.
  subroutine aero_binned_remove_particle_in_bin(aero_binned, bin_grid, &
       bin, comp_vol, aero_particle)

    !> Structure to remove the particle from.
    type(aero_binned_t), intent(inout) :: aero_binned
    !> Bin grid.
    type(bin_grid_t), intent(in) :: bin_grid
    !> Bin number of the aero_particle.
    integer, intent(in) :: bin
    !> Computational volume (m^3).
    real*8, intent(in) :: comp_vol
    !> Particle to remove.
    type(aero_particle_t), intent(in) :: aero_particle

    aero_binned%num_conc(bin) = aero_binned%num_conc(bin) &
         - 1d0 / comp_vol / bin_grid%dlnr
    aero_binned%vol_conc(bin,:) = aero_binned%vol_conc(bin,:) &
         - aero_particle%vol / comp_vol / bin_grid%dlnr

  end subroutine aero_binned_remove_particle_in_bin

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Update the aero_binned_t structure for the removal of the given
  !> particle.
  !!
  !! If the correct bin for the particle is already known then it is
  !! cheaper to call aero_binned_remove_particle_in_bin().
  subroutine aero_binned_remove_particle(aero_binned, bin_grid, &
       comp_vol, aero_particle)

    !> Structure to remove the particle from.
    type(aero_binned_t), intent(inout) :: aero_binned
    !> Bin grid.
    type(bin_grid_t), intent(in) :: bin_grid
    !> Computational volume (m^3).
    real*8, intent(in) :: comp_vol
    !> Particle to remove.
    type(aero_particle_t), intent(in) :: aero_particle

    call aero_binned_remove_particle_in_bin(aero_binned, bin_grid, &
         aero_particle_in_bin(aero_particle, bin_grid), comp_vol, &
         aero_particle)

  end subroutine aero_binned_remove_particle

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Add two aero_binned_t structures together.
  !!
  !! Symbolically does aero_binned = aero_binned + aero_binned_delta.
  subroutine aero_binned_add(aero_binned, aero_binned_delta)

    !> Base aero_binned_t structure that will be added to.
    type(aero_binned_t), intent(inout) :: aero_binned
    !> Structure to add to aero_binned.
    type(aero_binned_t), intent(in) :: aero_binned_delta

    aero_binned%num_conc = aero_binned%num_conc + aero_binned_delta%num_conc
    aero_binned%vol_conc = aero_binned%vol_conc + aero_binned_delta%vol_conc

  end subroutine aero_binned_add

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Subtract one aero_binned_t structure from another.
  !!
  !! Symbolically does aero_binned = aero_binned - aero_binned_delta.
  subroutine aero_binned_sub(aero_binned, aero_binned_delta)

    !> Base aero_binned_t structure that will be subtracted from.
    type(aero_binned_t), intent(inout) :: aero_binned
    !> Structure to subtract from aero_binned.
    type(aero_binned_t), intent(in) :: aero_binned_delta

    aero_binned%num_conc = aero_binned%num_conc - aero_binned_delta%num_conc
    aero_binned%vol_conc = aero_binned%vol_conc - aero_binned_delta%vol_conc

  end subroutine aero_binned_sub

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Scale an aero_binned_t by a real number.
  !!
  !! Symbolically does aero_binned = aero_binned * alpha.
  subroutine aero_binned_scale(aero_binned, alpha)

    !> Base aero_binned to scale.
    type(aero_binned_t), intent(inout) :: aero_binned
    !> Scale factor.
    real*8, intent(in) :: alpha

    aero_binned%num_conc = aero_binned%num_conc * alpha
    aero_binned%vol_conc = aero_binned%vol_conc * alpha

  end subroutine aero_binned_scale

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Copy one aero_binned_t structure to another.
  !!
  !! Symbolically does aero_binned_to = aero_binned_from.
  subroutine aero_binned_copy(aero_binned_from, aero_binned_to)

    !> Base aero_binned_t structure to copy from.
    type(aero_binned_t), intent(in) :: aero_binned_from
    !> Structure to copy to.
    type(aero_binned_t), intent(out) :: aero_binned_to

    integer :: n_bin, n_spec

    n_bin = size(aero_binned_from%vol_conc, 1)
    n_spec = size(aero_binned_from%vol_conc, 2)
    call aero_binned_deallocate(aero_binned_to)
    call aero_binned_allocate_size(aero_binned_to, n_bin, n_spec)
    aero_binned_to%num_conc = aero_binned_from%num_conc
    aero_binned_to%vol_conc = aero_binned_from%vol_conc

  end subroutine aero_binned_copy

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Add an aero_dist_t to an aero_binned_t.
  !!
  !! Symbolically does aero_binned = aero_binned + aero_dist.
  subroutine aero_binned_add_aero_dist(aero_binned, bin_grid, aero_data, &
       aero_dist)

    !> Base aero_binned_t structure to add to.
    type(aero_binned_t), intent(inout) :: aero_binned
    !> Bin grid.
    type(bin_grid_t), intent(in) :: bin_grid
    !> Aerosol data.
    type(aero_data_t), intent(in) :: aero_data
    !> The aero_dist_t structure to add.
    type(aero_dist_t), intent(in) :: aero_dist

    integer :: i_mode, i_spec
    real*8 :: mode_num_conc(bin_grid%n_bin)
    real*8 :: mode_vol_conc(bin_grid%n_bin)
    real*8 :: mode_vol_spec_den(bin_grid%n_bin, aero_data%n_spec)
    type(aero_mode_t), pointer :: aero_mode

    do i_mode = 1,aero_dist%n_mode
       aero_mode => aero_dist%mode(i_mode)
       if (aero_mode%type == "log_normal") then
          call num_conc_log_normal(aero_mode%mean_radius, &
               aero_mode%log10_std_dev_radius, bin_grid, mode_num_conc)
          call vol_conc_log_normal(aero_mode%mean_radius, &
               aero_mode%log10_std_dev_radius, bin_grid, mode_vol_conc)
       elseif (aero_mode%type == "exp") then
          call num_conc_exp(aero_mode%mean_radius, bin_grid, mode_num_conc)
          call vol_conc_exp(aero_mode%mean_radius, bin_grid, mode_vol_conc)
       elseif (aero_mode%type == "mono") then
          call num_conc_mono(aero_mode%mean_radius, bin_grid, mode_num_conc)
          call vol_conc_mono(aero_mode%mean_radius, bin_grid, mode_vol_conc)
       else
          call die_msg(749122931, "Unknown aero_mode type")
       end if
       mode_num_conc = mode_num_conc * aero_mode%num_conc
       mode_vol_conc = mode_vol_conc * aero_mode%num_conc
       do i_spec = 1,aero_data%n_spec
          mode_vol_spec_den(:,i_spec) = mode_vol_conc &
               * aero_mode%vol_frac(i_spec)
       end do
       aero_binned%num_conc = aero_binned%num_conc + mode_num_conc
       aero_binned%vol_conc = aero_binned%vol_conc + mode_vol_spec_den
    end do

  end subroutine aero_binned_add_aero_dist

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Determine the number of bytes required to pack the structure.
  !!
  !! See pmc_mpi for usage details.
  integer function pmc_mpi_pack_size_aero_binned(val)

    !> Structure to pack.
    type(aero_binned_t), intent(in) :: val

    pmc_mpi_pack_size_aero_binned = &
         pmc_mpi_pack_size_real_array(val%num_conc) &
         + pmc_mpi_pack_size_real_array_2d(val%vol_conc)

  end function pmc_mpi_pack_size_aero_binned

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Pack the structure into the buffer and advance position.
  !!
  !! See pmc_mpi for usage details.
  subroutine pmc_mpi_pack_aero_binned(buffer, position, val)

    !> Memory buffer.
    character, intent(inout) :: buffer(:)
    !> Current buffer position.
    integer, intent(inout) :: position
    !> Structure to pack.
    type(aero_binned_t), intent(in) :: val

#ifdef PMC_USE_MPI
    integer :: prev_position

    prev_position = position
    call pmc_mpi_pack_real_array(buffer, position, val%num_conc)
    call pmc_mpi_pack_real_array_2d(buffer, position, val%vol_conc)
    call assert(348207873, &
         position - prev_position == pmc_mpi_pack_size_aero_binned(val))
#endif

  end subroutine pmc_mpi_pack_aero_binned

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Unpack the structure from the buffer and advance position.
  !!
  !! See pmc_mpi for usage details.
  subroutine pmc_mpi_unpack_aero_binned(buffer, position, val)

    !> Memory buffer.
    character, intent(inout) :: buffer(:)
    !> Current buffer position.
    integer, intent(inout) :: position
    !> Structure to unpack into (must not be allocated).
    type(aero_binned_t), intent(out) :: val

#ifdef PMC_USE_MPI
    integer :: prev_position

    prev_position = position
    call pmc_mpi_unpack_real_array(buffer, position, val%num_conc)
    call pmc_mpi_unpack_real_array_2d(buffer, position, val%vol_conc)
    call assert(878267066, &
         position - prev_position == pmc_mpi_pack_size_aero_binned(val))
#endif

  end subroutine pmc_mpi_unpack_aero_binned

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Computes the average of the structure across all processors,
  !> storing the result on the root processor.
  subroutine pmc_mpi_reduce_avg_aero_binned(val, val_avg)

    !> Per-processor value to average.
    type(aero_binned_t), intent(in) :: val
    !> Averaged result (only valid on root processor).
    type(aero_binned_t), intent(out) :: val_avg

    call pmc_mpi_reduce_avg_real_array(val%num_conc, val_avg%num_conc)
    call pmc_mpi_reduce_avg_real_array_2d(val%vol_conc, val_avg%vol_conc)

  end subroutine pmc_mpi_reduce_avg_aero_binned

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Write full state.
  subroutine aero_binned_output_netcdf(aero_binned, ncid, bin_grid, &
       aero_data)
    
    !> Aero_binned to write.
    type(aero_binned_t), intent(in) :: aero_binned
    !> NetCDF file ID, in data mode.
    integer, intent(in) :: ncid
    !> bin_grid structure.
    type(bin_grid_t), intent(in) :: bin_grid
    !> aero_data structure.
    type(aero_data_t), intent(in) :: aero_data

    integer :: dimid_aero_radius, dimid_aero_species
    real*8 :: mass_den(bin_grid%n_bin, aero_data%n_spec)
    integer :: i_bin
    
    do i_bin = 1,bin_grid%n_bin
       mass_den(i_bin,:) = aero_binned%vol_conc(i_bin,:) &
            * aero_data%density
    end do

    call bin_grid_netcdf_dim_aero_radius(bin_grid, ncid, &
         dimid_aero_radius)
    call aero_data_netcdf_dim_aero_species(aero_data, ncid, &
         dimid_aero_species)

    call pmc_nc_write_real_1d(ncid, aero_binned%num_conc, &
         "aero_number_concentration", "1/m^3", (/ dimid_aero_radius /))
    call pmc_nc_write_real_2d(ncid, mass_den, &
         "aero_mass_concentration", "kg/m^3", &
         (/ dimid_aero_radius, dimid_aero_species /))

  end subroutine aero_binned_output_netcdf

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Read full state.
  subroutine aero_binned_input_netcdf(aero_binned, ncid, bin_grid, &
       aero_data)
    
    !> Aero_binned to write.
    type(aero_binned_t), intent(inout) :: aero_binned
    !> NetCDF file ID, in data mode.
    integer, intent(in) :: ncid
    !> bin_grid structure.
    type(bin_grid_t), intent(in) :: bin_grid
    !> aero_data structure.
    type(aero_data_t), intent(in) :: aero_data

    real*8 :: mass_den(bin_grid%n_bin, aero_data%n_spec)
    integer :: i_bin
    character(len=1000) :: unit

    call pmc_nc_read_real_1d(ncid, aero_binned%num_conc, &
         "aero_number_concentration", unit)
    call pmc_nc_read_real_2d(ncid, mass_den, &
         "aero_mass_concentration", unit)

    do i_bin = 1,bin_grid%n_bin
       aero_binned%vol_conc(i_bin,:) = mass_den(i_bin,:) &
            / aero_data%density
    end do

  end subroutine aero_binned_input_netcdf

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
end module pmc_aero_binned
