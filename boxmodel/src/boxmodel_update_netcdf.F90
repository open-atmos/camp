module boxmodel_update_netcdf
  use netcdf

  use camp_util, only: to_string
  use camp_constants, only: i_kind, dp
  use camp_camp_state, only: camp_state_t

  use boxmodel_io, only: ncdf_writer, handle_err
  use boxmodel_emissions, only: emissions_map_t

  use boxmodel_log
contains


  subroutine put_species_concentrations(this, camp_state, time_indx)
    class(ncdf_writer), intent(inout) :: this

    type(camp_state_t), intent(in) :: camp_state
    integer(kind=i_kind), intent(in)  :: time_indx

    integer(kind=i_kind) :: status, num_gas_specs
    ! aerosol dimensions
    integer(kind=i_kind) :: num_reps, n_cells
    integer(kind=i_kind), allocatable :: num_bins(:, :), num_sections(:), num_species(:, :)
    integer(kind=i_kind) :: max_sections, max_phases, max_bins, max_species
    integer(kind=i_kind) :: isect, irep, iphase, ibin, ispec, iaerrep, state_indx,i_cell

    integer(kind=i_kind) :: curr_phase_indx

    real(kind=dp), allocatable, dimension(:, :) :: bin_diameter

    real(kind=dp)        ::  aer_conc_value

    status = nf90_inquire_dimension(this%ncid, this%gas_species_dimid, len=num_gas_specs)
    call handle_err(218299941, status)
    status = nf90_inquire_dimension(this%ncid, this%aer_rep_dimid, len=num_reps)
    call handle_err(539464556, status)
    status = nf90_inquire_dimension(this%ncid, this%aer_section_dimid, len=max_sections)
    call handle_err(152849567, status)
    status = nf90_inquire_dimension(this%ncid, this%aer_phase_dimid, len=max_phases)
    call handle_err(352729362, status)
    status = nf90_inquire_dimension(this%ncid, this%bins_dimid, len=max_bins)
    call handle_err(722919763, status)
    status = nf90_inquire_dimension(this%ncid, this%aer_species_dimid, len=max_species)
    call handle_err(487537374, status)
    status = nf90_inquire_dimension(this%ncid, this%cells_dimid, len=n_cells)
    call handle_err(124941251, status)

    ! store the aerosol concentrations
    ! this%output_map contains the mapping between the state_var index
    ! and the multiple dimension output array
    !TODO: remap aerosol species printing for multiple cells
    do irep = 1, num_reps
      do isect = 1, this%num_sections(irep)
        do ibin = 1, this%num_bins(isect, irep)
          do iphase = 1, this%num_unique_phases(irep, isect)
            curr_phase_indx = this%phase_ids(irep, isect, iphase)
            do ispec = 1, size(this%output_map, 5)
              state_indx = this%output_map(irep, isect, ibin, curr_phase_indx, ispec)
              do i_cell = 1, n_cells
                if (state_indx > 0) then
                  aer_conc_value = camp_state%state_var(state_indx + ((i_cell - 1)*this%state_size_per_cell))
                else
                  aer_conc_value = 0.
                end if
                status = nf90_put_var( &
                  ncid=this%ncid, &
                  varid=this%aer_concentrations, &
                  values=(/aer_conc_value/), &
                  start=(/ispec, curr_phase_indx, ibin, isect, irep, i_cell, time_indx/), &
                  count=(/1, 1, 1, 1, 1, 1, 1/))
                call handle_err(126467935, status)
              enddo
            end do
          end do
        end do
      end do
    end do

    ! TODO: check that gas phase species are printed in the correct order.
    do i_cell = 1, n_cells
      do ispec = 1, num_gas_specs
        status = nf90_put_var( &
          ncid=this%ncid, &
          varid=this%gas_concentrations, &
          values=(/camp_state%state_var(ispec + ((i_cell - 1)*this%state_size_per_cell))/), &
          start=(/ispec, i_cell, time_indx/), &
          count=(/1, 1, 1/) &
          )
        call handle_err(339854477, status)
      enddo
    enddo

  end subroutine put_species_concentrations

  subroutine put_environment_variables(this, latitude, longitude, altitude, &
    pressure, temperature, humidity, time, time_indx, sza, height)
    class(ncdf_writer), intent(inout) :: this
    real(kind=dp), dimension(:), intent(in) :: latitude, longitude, altitude
    real(kind=dp), dimension(:), intent(in) :: pressure, temperature, humidity, sza, height
    real(kind=dp), intent(in) :: time
    integer(kind=i_kind), intent(in)  :: time_indx
    integer(kind=i_kind) :: n_cells

    integer(kind=i_kind) :: status

    status = nf90_inquire_dimension(this%ncid, this%cells_dimid, len=n_cells)
    call handle_err(124941251, status)

    ! put time
    call thread_log%debug("putting concentrations a time "//to_string(time))
    status = nf90_put_var( &
      ncid=this%ncid, &
      varid=this%time_varid, &
      values=(/time/), &
      start=(/time_indx/), &
      count=(/1/))
    call handle_err(848149432, status)

    ! put latitude
    status = nf90_put_var( &
      ncid=this%ncid, &
      varid=this%latitude_varid, &
      values=(/latitude/), &
      start=(/1, time_indx/), &
      count=(/n_cells, 1/))
    call handle_err(734542928, status)

    ! put latitude
    status = nf90_put_var( &
      ncid=this%ncid, &
      varid=this%longitude_varid, &
      values=(/longitude/), &
      start=(/1, time_indx/), &
      count=(/n_cells, 1/))
    call handle_err(954661977, status)

    ! put latitude
    status = nf90_put_var( &
      ncid=this%ncid, &
      varid=this%altitude_varid, &
      values=(/altitude/), &
      start=(/1, time_indx/), &
      count=(/n_cells, 1/))
    call handle_err(441056278, status)

    ! put pressure
    status = nf90_put_var( &
      ncid=this%ncid, &
      varid=this%pressure_varid, &
      values=(/pressure/), &
      start=(/1, time_indx/), &
      count=(/n_cells, 1/))
    call handle_err(803546133, status)

    ! put temperature
    status = nf90_put_var( &
      ncid=this%ncid, &
      varid=this%temperature_varid, &
      values=(/temperature/), &
      start=(/1, time_indx/), &
      count=(/n_cells, 1/))
    call handle_err(748773360, status)

    ! put humidity
    status = nf90_put_var( &
      ncid=this%ncid, &
      varid=this%humidity_varid, &
      values=(/humidity/), &
      start=(/1, time_indx/), &
      count=(/n_cells, 1/))
    call handle_err(906240013, status)

    ! put sza
    status = nf90_put_var( &
      ncid=this%ncid, &
      varid=this%sza_id, &
      values=(/sza/), &
      start=(/1, time_indx/), &
      count=(/n_cells, 1/))
    call handle_err(780978953, status)

    ! put height
    status = nf90_put_var( &
      ncid=this%ncid, &
      varid=this%height_id, &
      values=(/height/), &
      start=(/1, time_indx/), &
      count=(/n_cells, 1/))
    call handle_err(308369819, status)

  end subroutine put_environment_variables

  subroutine put_emission_rates(this, emissions_map, id_cell, time_indx)
    class(ncdf_writer), intent(inout) :: this
    class(emissions_map_t), intent(in) :: emissions_map
    integer(kind=i_kind), intent(in)  :: time_indx
    integer(kind=i_kind), intent(in)  :: id_cell

    integer(kind=i_kind) :: i_emis, status
    real(kind=dp) :: emis_value

    do i_emis = 1, emissions_map%n_emission

      emis_value = emissions_map%current_emissions_rates(i_emis)
      status = nf90_put_var( &
        ncid = this%ncid, &
        varid = this%emissions_varid, &
        values = (/emis_value/), &
        start = (/ emissions_map%emitted_id(i_emis), id_cell, time_indx/), &
        count = (/1, 1, 1/))
      call handle_err(841361582, status)

    end do
  end subroutine put_emission_rates

end module boxmodel_update_netcdf
