! Copyright (C) 2007-2021 Barcelona Supercomputing Center and University of
! Illinois at Urbana-Champaign
! SPDX-License-Identifier: MIT

!> \file
!> The camp_rand module.

!> Random number generators.
module camp_rand

  use camp_util
  use camp_constants
  use camp_mpi

  !> Length of a UUID string.
  integer, parameter :: CAMP_UUID_LEN = 36

  !> Next sequential id
  integer, private :: next_id = 100

contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Initializes the random number generator to the state defined by
  !> the given seed plus offset. If the seed is 0 then a seed is
  !> auto-generated from the current time plus offset.
  subroutine camp_srand(seed, offset)

    !> Random number generator seed.
    integer, intent(in) :: seed
    !> Random number generator offset.
    integer, intent(in) :: offset

    integer :: i, n, clock
    integer, allocatable :: seed_vec(:)

    if (seed == 0) then
       if (camp_mpi_rank() == 0) then
          call system_clock(count = clock)
       end if
       ! ensure all nodes use exactly the same seed base, to avoid
       ! accidental correlations
       call camp_mpi_bcast_integer(clock)
    else
       clock = seed
    end if
    clock = clock + 67 * offset
    call random_seed(size = n)
    allocate(seed_vec(n))
    i = 0 ! HACK to shut up gfortran warning
    seed_vec = clock + 37 * (/ (i - 1, i = 1, n) /)
    call random_seed(put = seed_vec)
    deallocate(seed_vec)

  end subroutine camp_srand

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Cleanup the random number generator.
  subroutine camp_rand_finalize()
  end subroutine camp_rand_finalize

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Returns a random number between 0 and 1.
  real(kind=dp) function camp_random()

    real(kind=dp) :: rnd

    call random_number(rnd)
    camp_random = rnd

  end function camp_random

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Returns a random integer between 1 and n.
  integer function camp_rand_int(n)

    !> Maximum random number to generate.
    integer, intent(in) :: n

    call assert(669532625, n >= 1)
    camp_rand_int = mod(int(camp_random() * real(n, kind=dp)), n) + 1
    call assert(515838689, camp_rand_int >= 1)
    call assert(802560153, camp_rand_int <= n)

  end function camp_rand_int

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Round val to \c floor(val) or \c ceiling(val) with probability
  !> proportional to the relative distance from \c val. That is,
  !> Prob(prob_round(val) == floor(val)) = ceil(val) - val.
  integer function prob_round(val)

    !> Value to round.
    real(kind=dp), intent(in) :: val

    ! FIXME: can replace this with:
    ! prob_round = floor(val + camp_random())
    if (camp_random() < real(ceiling(val), kind=dp) - val) then
       prob_round = floor(val)
    else
       prob_round = ceiling(val)
    endif

  end function prob_round

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Generate a Poisson-distributed random number with the given
  !> mean.
  !!
  !! See http://en.wikipedia.org/wiki/Poisson_distribution for
  !! information on the method. The method used at present is rather
  !! inefficient and inaccurate (brute force for mean below 10 and
  !! normal approximation above that point).
  !!
  !! The best known method appears to be due to Ahrens and Dieter (ACM
  !! Trans. Math. Software, 8(2), 163-179, 1982) and is available (in
  !! various forms) from:
  !!     - http://www.netlib.org/toms/599
  !!     - http://www.netlib.org/random/ranlib.f.tar.gz
  !!     - http://users.bigpond.net.au/amiller/random/random.f90
  !!     - http://www.netlib.org/random/random.f90
  !!
  !! Unfortunately the above code is under the non-free license:
  !!     - http://www.acm.org/pubs/copyright_policy/softwareCRnotice.html
  !!
  !! For other reasonable methods see L. Devroye, "Non-Uniform Random
  !! Variate Generation", Springer-Verlag, 1986.
  integer function rand_poisson(mean)

    !> Mean of the distribution.
    real(kind=dp), intent(in) :: mean
    real(kind=dp) :: L, p
    integer :: k

    call assert(368397056, mean >= 0d0)
    if (mean <= 10d0) then
       ! exact method due to Knuth
       L = exp(-mean)
       k = 0
       p = 1d0
       do
          k = k + 1
          p = p * camp_random()
          if (p < L) exit
       end do
       rand_poisson = k - 1
    else
       ! normal approximation with a continuity correction
       k = nint(rand_normal(mean, sqrt(mean)))
       rand_poisson = max(k, 0)
    end if

  end function rand_poisson

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Generate a Binomial-distributed random number with the given
  !> parameters.
  !!
  !! See http://en.wikipedia.org/wiki/Binomial_distribution for
  !! information on the method. The method used at present is rather
  !! inefficient and inaccurate (brute force for \f$np \le 10\f$ and
  !! \f$n(1-p) \le 10\f$, otherwise normal approximation).
  !!
  !! For better methods, see L. Devroye, "Non-Uniform Random Variate
  !! Generation", Springer-Verlag, 1986.
  integer function rand_binomial(n, p)

    !> Sample size.
    integer, intent(in) :: n
    !> Sample probability.
    real(kind=dp), intent(in) :: p
    real(kind=dp) :: np, nomp, q, G_real
    integer :: k, G, sum

    call assert(130699849, n >= 0)
    call assert(754379796, p >= 0d0)
    call assert(678506029, p <= 1d0)
    np = real(n, kind=dp) * p
    nomp = real(n, kind=dp) * (1d0 - p)
    if ((np >= 10d0) .and. (nomp >= 10d0)) then
       ! normal approximation with continuity correction
       k = nint(rand_normal(np, sqrt(np * (1d0 - p))))
       rand_binomial = min(max(k, 0), n)
    elseif (np < 1d-200) then
       rand_binomial = 0
    elseif (nomp < 1d-200) then
       rand_binomial = n
    else
       ! First waiting time method of Devroye (p. 525).
       ! We want p small, so if p > 1/2 then we compute with 1 - p and
       ! take n - k at the end.
       if (p <= 0.5d0) then
          q = p
       else
          q = 1d0 - p
       end if
       k = 0
       sum = 0
       do
          ! G is geometric(q)
          G_real = log(camp_random()) / log(1d0 - q)
          ! early bailout for cases to avoid integer overflow
          if (G_real > real(n - sum, kind=dp)) exit
          G = ceiling(G_real)
          sum = sum + G
          if (sum > n) exit
          k = k + 1
       end do
       if (p <= 0.5d0) then
          rand_binomial = k
       else
          rand_binomial = n - k
       end if
       call assert(359087410, rand_binomial <= n)
    end if

  end function rand_binomial

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Generates a normally distributed random number with the given
  !> mean and standard deviation.
  real(kind=dp) function rand_normal(mean, stddev)

    !> Mean of distribution.
    real(kind=dp), intent(in) :: mean
    !> Standard deviation of distribution.
    real(kind=dp), intent(in) :: stddev
    real(kind=dp) :: u1, u2, r, theta, z0, z1

    call assert(898978929, stddev >= 0d0)
    ! Uses the Box-Muller transform
    ! http://en.wikipedia.org/wiki/Box-Muller_transform
    u1 = camp_random()
    u2 = camp_random()
    r = sqrt(-2d0 * log(u1))
    theta = 2d0 * const%pi * u2
    z0 = r * cos(theta)
    z1 = r * sin(theta)
    ! z0 and z1 are now independent N(0,1) random variables
    ! We throw away z1, but we could use a SAVE variable to only do
    ! the computation on every second call of this function.
    rand_normal = stddev * z0 + mean

  end function rand_normal

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Generates a vector of normally distributed random numbers with
  !> the given means and standard deviations. This is a set of
  !> normally distributed scalars, not a normally distributed vector.
  subroutine rand_normal_array_1d(mean, stddev, val)

    !> Array of means.
    real(kind=dp), intent(in) :: mean(:)
    !> Array of standard deviations.
    real(kind=dp), intent(in) :: stddev(size(mean))
    !> Array of sampled normal random variables.
    real(kind=dp), intent(out) :: val(size(mean))

    integer :: i

    do i = 1,size(mean)
       val(i) = rand_normal(mean(i), stddev(i))
    end do

  end subroutine rand_normal_array_1d

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Sample the given continuous probability density function.
  !!
  !! That is, return a number k = 1,...,n such that prob(k) = pdf(k) /
  !! sum(pdf). Uses accept-reject.
  integer function sample_cts_pdf(pdf)

    !> Probability density function (not normalized).
    real(kind=dp), intent(in) :: pdf(:)

    real(kind=dp) :: pdf_max
    integer :: k
    logical :: found

    ! use accept-reject
    pdf_max = maxval(pdf)
    if (minval(pdf) < 0d0) then
       call die_msg(121838078, 'pdf contains negative values')
    end if
    if (pdf_max <= 0d0) then
       call die_msg(119208863, 'pdf is not positive')
    end if
    found = .false.
    do while (.not. found)
       k = camp_rand_int(size(pdf))
       if (camp_random() < pdf(k) / pdf_max) then
          found = .true.
       end if
    end do
    sample_cts_pdf = k

  end function sample_cts_pdf

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Sample the given discrete probability density function.
  !!
  !! That is, return a number k = 1,...,n such that prob(k) = pdf(k) /
  !! sum(pdf). Uses accept-reject.
  integer function sample_disc_pdf(pdf)

    !> Probability density function.
    integer, intent(in) :: pdf(:)

    integer :: pdf_max, k
    logical :: found

    ! use accept-reject
    pdf_max = maxval(pdf)
    if (minval(pdf) < 0) then
       call die_msg(598024763, 'pdf contains negative values')
    end if
    if (pdf_max <= 0) then
       call die_msg(109961454, 'pdf is not positive')
    end if
    found = .false.
    do while (.not. found)
       k = camp_rand_int(size(pdf))
       if (camp_random() < real(pdf(k), kind=dp) / real(pdf_max, kind=dp)) then
          found = .true.
       end if
    end do
    sample_disc_pdf = k

  end function sample_disc_pdf

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Convert a real-valued vector into an integer-valued vector by
  !> sampling.
  !!
  !! Use n_samp samples. Each discrete entry is sampled with a PDF
  !! given by vec_cts. This is very slow for large n_samp or large n.
  subroutine sample_vec_cts_to_disc(vec_cts, n_samp, vec_disc)

    !> Continuous vector.
    real(kind=dp), intent(in) :: vec_cts(:)
    !> Number of discrete samples to use.
    integer, intent(in) :: n_samp
    !> Discretized vector.
    integer, intent(out) :: vec_disc(size(vec_cts))

    integer :: i_samp, k

    call assert(617770167, n_samp >= 0)
    vec_disc = 0
    do i_samp = 1,n_samp
       k = sample_cts_pdf(vec_cts)
       vec_disc(k) = vec_disc(k) + 1
    end do

  end subroutine sample_vec_cts_to_disc

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Generate a random hexadecimal character.
  character function rand_hex_char()

    integer :: i

    i = camp_rand_int(16)
    if (i <= 10) then
       rand_hex_char = achar(iachar('0') + i - 1)
    else
       rand_hex_char = achar(iachar('A') + i - 11)
    end if

  end function rand_hex_char

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Generate a version 4 UUID as a string.
  !!
  !! See http://en.wikipedia.org/wiki/Universally_Unique_Identifier
  !! for format details.
  subroutine uuid4_str(uuid)

    !> The newly generated UUID string.
    character(len=CAMP_UUID_LEN), intent(out) :: uuid

    integer :: i

    do i = 1,8
       uuid(i:i) = rand_hex_char()
    end do
    uuid(9:9) = '-'
    do i = 1,4
       uuid((i + 9):(i + 9)) = rand_hex_char()
    end do
    uuid(14:14) = '-'
    do i = 1,4
       uuid((i + 14):(i + 14)) = rand_hex_char()
    end do
    uuid(19:19) = '-'
    do i = 1,4
       uuid((i + 19):(i + 19)) = rand_hex_char()
    end do
    uuid(24:24) = '-'
    do i = 1,12
       uuid((i + 24):(i + 24)) = rand_hex_char()
    end do

    uuid(15:15) = '4'

    i = camp_rand_int(4)
    if (i <= 2) then
       uuid(20:20) = achar(iachar('8') + i - 1)
    else
       uuid(20:20) = achar(iachar('A') + i - 3)
    end if

  end subroutine uuid4_str

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  !> Generate an integer id
  !! Ids will be sequential, and can only be generated by the primary process
  function generate_int_id() result(id)

    !> The generated id
    integer(kind=i_kind) :: id

    call assert_msg(226665912, camp_mpi_rank().eq.0, &
                    "Sequential ids can only be generated on the primary "//&
                    "process.")

    id = next_id
    next_id = next_id + 1

  end function generate_int_id

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

end module camp_rand
