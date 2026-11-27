!-------------------------------------------------------------------------------
! File: spam_runs.f90
! Author: Matthew Ogden
! Email: ogdenm12@gmail.com
! Github: mbogden
! Created: 2024-Apr-17
! Compiler: gfortran, f2py
! 
! Description:
! This module is designed to be a wrapper for the SPAM simulation, handling the
! running of various simulations and returning the results to spam_wrapper.py
!
! References:
! Ssections of this code were written with the assistance of ChatGPT made by OpenAI.
!-------------------------------------------------------------------------------

subroutine multiply_and_add(a, b, product, sum)


    double precision, intent(in) :: a, b
    double precision, intent(out) :: product, sum

    product = a * b
    sum = a + b

end subroutine multiply_and_add


  
subroutine create_disks( npart_1, npart_2, my_array )

    use init_module
    use parameters_module
    use setup_module
    use io_module
    use integrator


    implicit none

    real (kind=8) :: t0, time_interval
    integer (kind=4) :: nstep_local
    real (kind=8) :: rrr
    integer :: nparticles1, nparticles2, npart_1, npart_2
    real(kind=8), intent(inout), dimension(:, :) :: my_array

    !------------------------------------------------------

    call RANDOM_SEED()

    ! set the target parameters
    nparticles1 = 1000
    nparticles2 = 500
    call DEFAULT_PARAMETERS(nparticles1, nparticles2)

    ! Sections of the following were created with chatGTP.
    ! You might still need some initialization for distribution
    call INIT_DISTRIBUTION

    ! create the disks
    call PROFILE(rin1, rout1, rscale1, 1, n1, mass1, eps1, &
        theta1, phi1, opt1, heat1, x0)
    call PROFILE(rin2, rout2, rscale2, n1+1, n, mass2, eps2, &
        theta2, phi2, opt2, heat2, x0)

    ! Copy particles to be sent back to spam_wrapper.py
    print *, 'Size of x0: ', size(x0)
    my_array = x0

end subroutine create_disks

subroutine set_encounter_parameters( py_args, n_args)
    implicit none
    character(len=50), dimension(:), intent(in) :: py_args
    integer, intent(in) :: n_args
    integer :: i

    ! Print arguments for sanity check
    print * , 'Number of arguments:', n_args
    do i = 1, n_args
        print *, 'Argument', i, ':', trim(adjustl(py_args(i)))
    end do

    ! Implement logic as needed
end subroutine set_encounter_parameters

subroutine basic_run()

    use init_module
    use parameters_module
    use setup_module
    use io_module
    use integrator


    implicit none

    character (len=20):: target_name
    real (kind=8) :: t0, time_interval
    integer (kind=4) :: nstep_local
    real (kind=8) :: rrr
    integer :: nparticles1, nparticles2
    !------------------------------------------------------

    ! set the disk parameters
    call RANDOM_SEED()

    ! ! set the target parameters
    nparticles1 = 1000!0!100000!10000
    nparticles2 = 500!0!50000!5000
    call DEFAULT_PARAMETERS(nparticles1, nparticles2)


    ! call CREATE_COLLISION

    ! !
    ! !     -----loop over the system for the output
    ! !

    ! ! initialize rk routine
    ! call INIT_RKVAR(x0, mass1, mass2, epsilon1, epsilon2, theta1, phi1, &
    !         theta2, phi2, rscale1, rscale2, rout1, rout2, n, n1, n2)


    ! t0 = tstart

    ! nstep = int( (tend - t0) / h) + 2
    ! nstep_local = nstep 

    ! time_interval = (tend - t0) * 2

    ! nunit = 50
    ! call OCTAVE_PARAMETERS_OUT(mass1, theta1, phi1, rout1, mass2, &
    !         theta2, phi2, rout2, original_rv(1:3), original_rv(4:6), time_interval, x0(n,:), n, nunit)

    ! !      call CREATE_IMAGES

    ! fname = trim(outfilename) // "_" // trim(distinguisher) // ".000"
    ! open(unit, file=trim(fname))
    ! call OUTPUT_PARTICLES(unit, x0, mass1, mass2, &
    !         eps1, eps2, &
    !         n, n1, n2, &
    !         time, header_on)
    ! close(unit)


    ! ! main integration loop
    ! iout = 0
    ! do istep = 1, nstep_local
    !     call TAKE_A_STEP
    !     rrr = sqrt(x0(n,1)*x0(n,1)+x0(n,2)*x0(n,2) + x0(n,3)*x0(n,3))

    !     if (mod(istep, showsteps) == 0) then
    ! !       print*,"q",istep, time, rrr
    !     endif



    !     if (mod(istep, 50) == 5 .and. show_all_steps) then
    !         call CREATE_IMAGES
    !     endif
    ! enddo

    ! !      call CREATE_IMAGES
    ! fname = trim(outfilename) // "_" // trim(distinguisher) // ".101"
    ! open(unit, file=trim(fname))
    ! call OUTPUT_PARTICLES(unit, x0, mass1, mass2, &
    !         eps1, eps2, &
    !         n, n1, n2, &
    !         time, header_on)
    ! close(unit)

    ! ! this creates a simple script for animating the output with gnuplot
    ! ! gnuplot
    ! ! i = 1
    ! ! j = 2
    ! ! load 'gscript
    ! if (.not. header_on) then
    !     call CREATE_GNUPLOT_SCRIPT(x0, iout)
    ! endif

    ! ! clean up memory
    ! deallocate(x0)
    ! deallocate(xout)
    ! call DEALLOCATE_RKVAR


    ! enddo


end subroutine basic_run
