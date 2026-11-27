module custom_runs_module

	use init_module
	use parameters_module
	use setup_module
	use io_module
	use integrator
	use df_module

    implicit none

    integer (kind=4) :: pre_setup = 0
    real(kind=8) :: rchalo, mhalo, mbulge, hbulge, hdisk
    
    contains

    subroutine basic_run(collision_param, npts1, npts2, h1, h2, init_pts, final_pts)
        
        real(kind=8), intent(in), dimension(22) :: collision_param
        integer, intent(in) :: npts1, npts2
        real(kind=8), intent(in) :: h1, h2
        
        real(kind=8), intent(out), dimension(npts1+npts2+1,6) :: init_pts, final_pts

        real (kind=8) :: t0, time_interval
        integer (kind=4) :: nstep_local
        real (kind=8) :: rrr
        integer :: nparticles1, nparticles2

        ! print *, "FR: Basic Run!"

        ! if pre_setup is 0, do the setup
        if (pre_setup == 0) then
            call SIMR_INIT()
            pre_setup = 1
        endif
        
        ! Define collision parameters
        call SIMR_SETUP_CUSTOM_COLLISION(collision_param, npts1, npts2, h1, h2)
    
        ! initialize dynamical friction, force/potential/acceleration profiles around galaxies, 
        call SIMR_CREATE_COLLISION()
            
        ! ! Saving initial particles
        ! copy the initial particles
        init_pts = x0

        ! Perform integration/perturbation of secondary's orbit
        call INIT_RKVAR(x0, mass1, mass2, epsilon1, epsilon2, theta1, phi1, &
            theta2, phi2, rscale1, rscale2, rout1, rout2, n, n1, n2)

        ! tstart/t0 is time of closest approach within orbit
        t0 = tstart

        ! Define total steps, 
        ! Equal steps before and after time of closest approach.
        nstep = int( (tend - t0) / h) + 2
        nstep_local = nstep 

        ! time_interval is the total time of the simulation. 
        time_interval = (tend - t0) * 2

        nunit = 50

        ! This writes parameters to disk "fort.50"

        ! main integration loop
        iout = 0
        do istep = 1, nstep_local
            call TAKE_A_STEP
            rrr = sqrt(x0(n,1)*x0(n,1)+x0(n,2)*x0(n,2) + x0(n,3)*x0(n,3))

            if (mod(istep, showsteps) == 0) then
        !       print*,"q",istep, time, rrr
            endif

            if (mod(istep, 50) == 5 .and. show_all_steps) then
                call CREATE_IMAGES
            endif
        enddo

        ! Save final pts
        final_pts = x0

        ! Deallocate the memory
        deallocate(x0)
        deallocate(xout)
        deallocate(projected)
        call deallocate_rkvar()
        call SIMR_CLEANUP()

        return
        
    end subroutine basic_run

    subroutine orbit_run(collision_param, in_n_steps, orbit_path)
        
        real(kind=8), intent(in), dimension(22) :: collision_param
        integer, intent(in) :: in_n_steps
        real(kind=8), intent(out), dimension(in_n_steps,7) :: orbit_path

        real (kind=8) :: t0, time_interval, current_time
        integer (kind=4) :: nstep_local
        real (kind=8) :: rrr
        integer :: nparticles1, nparticles2

        ! print *, "FR: Orbit Run!", " lnl: ", lnl_in

        ! if pre_setup is 0, do the setup
        if (pre_setup == 0) then
            call SIMR_INIT()
            pre_setup = 1
        endif
        
        ! Define collision parameters
        call SIMR_SETUP_CUSTOM_COLLISION(collision_param, 10, 5, 0.0d0, 0.0d0)
    
        ! Initialize the collision
        call SIMR_CREATE_COLLISION()
            
        ! initialize rk routine for orbit prediction
        call INIT_RKVAR(x0, mass1, mass2, epsilon1, epsilon2, theta1, phi1, &
            theta2, phi2, rscale1, rscale2, rout1, rout2, n, n1, n2)

        t0 = tstart
        time_interval = (tend - t0) * 2
        nstep = int( (tend - t0) / h) + 2
        nstep_local = nstep 
        nunit = 50


        ! main integration loop
        iout = 0
        current_time = 0.0d0
        do istep = 1, nstep_local
            call TAKE_A_STEP
            rrr = sqrt(x0(n,1)*x0(n,1)+x0(n,2)*x0(n,2) + x0(n,3)*x0(n,3))

            orbit_path(istep,1:6) = x0(n,1:6)
            orbit_path(istep,7) = current_time
            current_time = current_time + h

        enddo

        ! Deallocate the memory
        deallocate(x0)
        deallocate(xout)
        deallocate(projected)
        call deallocate_rkvar()
        call SIMR_CLEANUP()

        return
        
    end subroutine orbit_run

    subroutine calc_orbit_integration_steps( collision_param, out_n_steps )
        real(kind=8), intent(in), dimension(22) :: collision_param
        integer, intent(out) :: out_n_steps

        real (kind=8) :: t0, time_interval
        integer (kind=4) :: nstep_local
        real (kind=8) :: rrr

        ! if pre_setup is 0, do the setup
        if (pre_setup == 0) then
            call SIMR_INIT()
            pre_setup = 1
        endif

        ! Define collision parameters
        call SIMR_SETUP_CUSTOM_COLLISION(collision_param, 10, 5, 0.0d0, 0.0d0)

        ! print *, "FR: Creating disk!"
        call SIMR_CREATE_COLLISION()

        ! initialize rk routine for particle integration/perturbation
        call INIT_RKVAR(x0, mass1, mass2, epsilon1, epsilon2, theta1, phi1, &
            theta2, phi2, rscale1, rscale2, rout1, rout2, n, n1, n2)

        t0 = tstart

        nstep = int( (tend - t0) / h) + 2

        out_n_steps = nstep

        deallocate(x0)
        deallocate(xout)
        deallocate(projected)
        call deallocate_rkvar()
        call SIMR_CLEANUP()

        return

    end subroutine calc_orbit_integration_steps


    subroutine basic_disk(collision_param, npts1, npts2, init_pts)
        integer, intent(in) :: npts1, npts2
        real(kind=8), intent(in), dimension(22) :: collision_param
        real(kind=8), intent(out), dimension(npts1+npts2+1,6) :: init_pts

        ! print *, "FR: basic_disk!"

        ! if pre_setup is 0, do the setup
        if (pre_setup == 0) then
            call SIMR_INIT()
            pre_setup = 1
        endif

        ! Define collision parameters
        call SIMR_SETUP_CUSTOM_COLLISION(collision_param, npts1, npts2, 0.0d0, 0.0d0)
    
        ! print *, "FR: Creating disk!"
        call SIMR_CREATE_COLLISION()

        ! ! Output particles to disk
        init_pts = x0

        ! Deallocate the memory
        deallocate(x0)
        deallocate(xout)
        deallocate(projected)
        return

    end subroutine basic_disk

    subroutine testing_position_velocity(collision_param, nsteps, orbit_path)
        
        integer, intent(in) :: nsteps
        real(kind=8), intent(in), dimension(22) :: collision_param
        real(kind=8), intent(out), dimension(nsteps,7) :: orbit_path

        real (kind=8) :: t0, time_interval, current_time
        real(kind=8), dimension(3) :: current_pos, current_vel
        integer (kind=4) :: nstep_local
        real (kind=8) :: rrr
        integer :: nparticles1, nparticles2

        print *, "FR: Testing Position and Velocity!"

        ! if pre_setup is 0, do the setup
        if (pre_setup == 0) then
            call SIMR_INIT()
            pre_setup = 1
        endif
        
        ! Define collision parameters
        ! call SIMR_SETUP_CUSTOM_COLLISION(collision_param, 10, 5, 0.0d0, 0.0d0)
    
        t0 = 0
        time_interval = nsteps
        nstep = nsteps
        nstep_local = nsteps

        current_pos = collision_param(1:3)
        current_vel = collision_param(4:6)
        current_time = 0.0d0

        ! Going backwards in time
        iout = 0
        current_time = 0.0d0
        do istep = 1, nstep_local
            orbit_path(istep,1:3) = current_pos
            orbit_path(istep,4:6) = current_vel
            orbit_path(istep,7) = current_time

            current_pos = current_pos - current_vel * h
            current_time = current_time - h
        enddo

        ! Deallocate the memory
        ! deallocate(x0)
        ! deallocate(xout)
        ! deallocate(projected)
        ! call deallocate_rkvar()
        ! call SIMR_CLEANUP()

        return
        
    end subroutine testing_position_velocity

    subroutine SIMR_SETUP_CUSTOM_COLLISION(collision_param, npts1, npts2, h1, h2)

        implicit none

        ! Variable declarations
        integer, intent(in) :: npts1, npts2
        real(kind=8), intent(in) :: h1, h2
        real(kind=8), intent(in), dimension(22) :: collision_param

        ! Just saying hi!
        ! print *, "FR: SIMR_SETUP_CUSTOM_COLLISION!"

        ! Setup custom collision
        ! ! call CREATE_COLLISION
        ! ! call INIT_DISTRIBUTION

        ! print
        ! print *, "FR: Setting up custom collision"

        ! Hard coding potential type
        ! potential_type = 0  ! Default
        potential_type = 1 ! Only type to use dynamical friction

        ! ! set the collision parameters


        sec_vec(1) = collision_param(1)
        sec_vec(2) = collision_param(2)
        sec_vec(3) = collision_param(3)
        sec_vec(4) = collision_param(4)
        sec_vec(5) = collision_param(5)
        sec_vec(6) = collision_param(6)
        mass1 = collision_param(7)
        mass2 = collision_param(8)
        rout1 = collision_param(9)
        rout2 = collision_param(10)
        phi1 = collision_param(11)
        phi2 = collision_param(12)
        theta1 = collision_param(13)
        theta2 = collision_param(14)
        epsilon1 = collision_param(15)
        epsilon2 = collision_param(16)
        rscale1(1) = collision_param(17)
        rscale1(2) = collision_param(18)
        rscale1(3) = collision_param(19)
        rscale2(1) = collision_param(20)
        rscale2(2) = collision_param(21)
        rscale2(3) = collision_param(22)
        use_sec_vec = .true. 

        ! ! Derived values
        eps1 = epsilon1 * epsilon1
        eps2 = epsilon2 * epsilon2

        ! Define simulation parameters
        n1 = npts1
        n2 = npts2
        n = n1 + n2

        ! Allocate the space for the particles - n+1 here ONLY
        allocate(x0(n+1,6), stat=iostat)
        allocate(xout(n+1,6), stat=iostat)
        allocate(projected(n+1,3), stat=iostat)
        ! print *, "FR: n1, n2, n", n1, n2, n
        ! print *, "FR: Points allocated!"

        heat1 = h1
        heat2 = h2

        ! Hardcoded to always calculate how far back to go.
        tStart = 0.0d0  ! testing something
        tStart = -1000.0d0
        time = tStart
        tIsSet = .false.


    ! set the default collision  - testing only

    end subroutine SIMR_SETUP_CUSTOM_COLLISION

    subroutine SIMR_CREATE_COLLISION()

        implicit none
        real (kind=8), dimension(7) :: rv4min
        real (kind=8), dimension(4) :: tminVals
        real (kind=8) :: tmpT

        ! write print statment saying hi from function
        ! print *, "FR: SIMR_CREATE_COLLISION!"
            
        ! create the disks
      !  call SET_DIFFQ2_PARAMETERS(phi1, theta1, phi2, theta2, rscale1, rscale2, rout1, rout2)
      
        call PROFILE(rin1, rout1, rscale1, 1, n1, mass1, eps1, &
             theta1, phi1, opt1, heat1, x0)
      
        call PROFILE(rin2, rout2, rscale2, n1+1, n, mass2, eps2, &
             theta2, phi2, opt2, heat2, x0)
      
        ! determine if we need to calculate tStart
        if( .NOT. tIsSet ) then
          rv4min = (/sec_vec(1), sec_vec(2), sec_vec(3), -sec_vec(4), -sec_vec(5), -sec_vec(6), 0.0d0/)
          tminVals = getTStart(rv4min, mass1, mass2, sqrt(eps1), sqrt(eps2) , h,-30.0d0 ,10.0d0*(rout1), rout1, rout2)
      
          tmpT = tminVals(1)
          if ( tmpT < -12.0 ) then
            tmpT = -5
          endif
      
          if ( abs(tmpT) < h) then
            tmpT = -5
          endif
              tstart = tmpT
                      time = tstart
           tIsSet = .true.
        endif
      
        ! set the perturber galaxy position
        if( use_sec_vec ) then
          call PERTURBER_POSITION_VEC(sec_vec, mass1, mass2, eps1, eps2, h, n, n1, time, x0, original_rv)
        else
          call PERTURBER_POSITION(inclination_degree, omega_degree, rmin, &
               velocity_factor, &
               mass1, mass2, eps1, eps2, h, n, n1, time, x0, original_rv)
        endif

        return
      
      end subroutine SIMR_CREATE_COLLISION

    subroutine SIMR_INIT()

        implicit none

        integer :: nparticles1, nparticles2
        
        ! print *, "FR: SIMR_INIT!"

        ! Setup default parameters for a quick test
        nparticles1 = 100
        nparticles2 = 50
        call RANDOM_SEED()
        call STANDARD_GALAXY1( mass1, epsilon1, rin1, rout1, rscale1, theta1, phi1, opt1, heat1 )
        call STANDARD_GALAXY2(  mass2, epsilon2, rin2, rout2, rscale2, theta2, phi2, opt2, heat2 )

        ! Default values based on Milky Way and M31.
        call SIMR_INIT_DISTRIBUTION( 0.01d0, 10.0d0, 5.8d0, 0.3333d0, 2.0d0, 1.0d0)
        
        n1 = nparticles1
        n2 = nparticles2
        n = n1 + n2 

        call TEST_COLLISION(n, n1, n2, time, inclination_degree, omega_degree, &
          rmin, velocity_factor, h, nstep, nout) 
        
        pre_setup = 1 ! Let everyone know we are already setup

    end subroutine SIMR_INIT

  subroutine GET_SPAM_PARAMETERS(lnl_out, rchalo_out, mhalo_out, mbulge_out, hbulge_out, hdisk_out)
  ! subroutine GET_SPAM_PARAMETERS(lnl_out)

    implicit none
    real (kind=8), intent(out) :: lnl_out, rchalo_out, mhalo_out, mbulge_out, hbulge_out, hdisk_out
    ! real (kind=8), intent(out) :: lnl_out
    ! real (kind=8) :: lnl, rchalo, mhalo, mbulge, hbulge, hdisk
    ! print *, "FR: GET_SPAM_PARAMETERS!"
    lnl_out = lnl
    rchalo_out = rchalo
    mhalo_out = mhalo
    mbulge_out = mbulge
    hbulge_out = hbulge
    hdisk_out = hdisk

    return

  end subroutine GET_SPAM_PARAMETERS
      
  subroutine SIMR_INIT_DISTRIBUTION(lnl_in, rchalo_in, mhalo_in, mbulge_in, hbulge_in, hdisk_in)
    !     -----Description: initializes the distribution 
    !          on input:  
    !          on output:
    !     ----------------------------------------------------------------
        implicit none
    !     ----- Variable declarations ---------------------
        real (kind=8) :: lnl_in, rchalo_in, mhalo_in, mbulge_in, hbulge_in, hdisk_in
        real (kind=8) :: rmax
        real (kind=8) :: mold, dmold, mtot
        real (kind=8) :: dx, x
        ! real (kind=8) :: alphahalo, qhalo, gammahalo, mhalo, rchalo, rhalo, epsilon_halo
        real (kind=8) :: alphahalo, qhalo, gammahalo, rhalo, epsilon_halo
        ! real (kind=8) :: zdisk, hdisk, zdiskmax
        real (kind=8) :: zdisk, zdiskmax
        ! real (kind=8) :: hbulge, mbulge
        real (kind=8) :: rho_tmp
        real (kind=8) :: G, factor
        real (kind=8) :: r, m, sqrtpi
        real (kind=8) :: p1, rd, rho_local
        real (kind=8) :: p, rr, dr, rh, dp, mnew, dm
        real (kind=8) :: acc_merge, rad_merge, acc
    
        integer (kind=4) :: j, nmax, k, nmerge, ntotal, jj
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3

    ! print hello from function
        ! print *, "FR: SIMR_INIT_DISTRIBUTION!"
    ! Print function and lnl_in value
        ! print *, "FR: init_distribution.lnl_in: ", lnl_in
    
    !!!!!
    ! set the constant for dynamical friction
        lnl = 0.00d0
    ! default for Merger Zoo
        lnl = 0.001d0
    
    !!!!!
    ! set up the parameters for the halo
        mhalo = 5.8d0  
        rhalo = 10.0d0
        rchalo = 10.0d0
        gammahalo = 1.0d0
        epsilon_halo = 0.4d0 
        SqrtPI = sqrt(pi)

    ! Matt O.  Redefining Varibles as inputs
        lnl = lnl_in
        rchalo = rchalo_in
        mhalo = mhalo_in
        mbulge = mbulge_in
        hbulge = hbulge_in
        hdisk = hdisk_in
    
    !!!!!
    ! derive additional constants
        qhalo = gammahalo / rchalo
        alphahalo = 1.0d0 / ( 1.0d0 - SqrtPI * qhalo * exp(qhalo**2) * (1.0d0 - erf(qhalo)) )
    
    !!!!!
    ! set the integration limits and zero integration constants
        rmax = 20
        nmax = 2000
        dr = rmax / (nmax)
        mold = 0
    
        ntotal = nnn
    
    !!!!!
    ! set the limits for integration, and zero integration constants
        k = nmax / 2
        dx = 1.0  / k
        x = 0.0d0
        dmold = 0.0d0
        mtot = 0.0d0
        rad = 0.0d0
        m = 0.0d0
        G = 1
        
    
    !!!!!
    ! set the fundamental disk parameters
        zdisk = 0.2
        zdiskmax = 3.5
        hdisk = 1.0
        
    
    !!!!!
    ! set the fundamental bulge parameters
        hbulge = 0.2
        mbulge = 0.3333
        
        
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3
    !!!!! set up the radius array
        do j = 1, nmax
          x = x + dx
          rad(j)= x * rchalo
        end do
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3
    !!!!!
     
       dr = rad(2) - rad(1)
       dx = dr / rchalo 
       
       do j = 1, nmax
    
    ! set the position
          r = rad(j)
          x = r / rchalo
    
    ! calculate the local rho based 
          rho_tmp = alphahalo / (2*SqrtPI**3 ) * (exp(-x**2) / (x**2 + qhalo**2))
    
    ! renormalize for the new halo size
          rho_tmp = rho_tmp / ( rchalo * rchalo * rchalo) 
    
    ! calculate mass in local shell, and update total mass
    !      dm = rho_tmp * 4 * pi * x * x *dx
          dm = rho_tmp * 4 * pi * r * r *dr
          mtot = mtot + dm
    
    ! store values in an array
          rho_halo(j) = rho_tmp * mhalo 
          mass_halo(j) = mtot * mhalo
        end do
    
    !!!!!
    ! now calculate the potential
        do j = 1, nmax
          r = rad(j)
          m = mass_halo(j)
          p1 = -G * m / r
          phi(j) = p1
    
        end do
    
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3
    ! disk model    !!!!!
    ! loop over the distribution
        do j = 1,nmax
    
    ! set the radius
          rd = rad(j)
      
    ! find the local density in the disk
          rho_local  = exp(-rd/hdisk)/ (8*pi*hdisk**2.0d0) 
          rho_disk(j) = rho_local
          
    ! find the mass in the spherical shell
          mnew = 4 * pi * rho_local * rd *rd * dr
          
          mass_disk(j) = mnew + mold
          mold = mass_disk(j)
        end do
    

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!3
    ! bulge model    !!!!!
    ! loop over the distribution
        mold = 0.0
        do j = 1,nmax
    ! set the radius
          rd = rad(j)
      
    ! find the local density in the disk
          rho_local  = exp(-rd**2/hbulge**2)
          rho_bulge(j) = rho_local
    
    ! find the mass in the spherical shell
          mnew = 4 * pi * rho_local * rd *rd * dr
          
          mass_bulge(j) = mnew + mold
          mold = mass_bulge(j)
        end do
    
    ! renormalize distribution
        factor = mbulge / mass_bulge(nmax)
        do j = 1,nmax
          mass_bulge(j) = mass_bulge(j) * factor
          rho_bulge(j)  = rho_bulge(j)  * factor
        end do
    
      
        dr = rad(2) - rad(1)      
       
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        j = 1
        mass_total(j)=  (mass_halo(j) + mass_disk(j) + mass_bulge(j)) 
        r = rad(j)
        rho_total(j) = mass_total(j) /  (4.0d0/3.0d0 * pi * r * r * r)
        dr = rad(2) - rad(1)
    
        do j = 2,nmax
          r = rad(j)
          mass_total(j)=  (mass_halo(j) + mass_disk(j) + mass_bulge(j)) 
    
          dm = mass_total(j) - mass_total(j-1)
          rho_total(j) = dm / (4 * pi * r * r * dr)
    
        end do
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! find the velocity dispersion v_r**2
    
        masses = mass_total
        radius = rad
        density = rho_total
    
    
        do j = 1,nmax
    
          p = 0.0d0
          rr = radius(j)
          dr = radius(nmax) / nmax
          do jj = j,nmax
          
            m  = masses(jj)
            rh = density(jj)
            rr = rr + dr
            
            dp = rh * G * m / rr**2 * dr
            p = p + dp
          end do
          
          vr2(j) = 1/density(j) * p
          vr(j) = sqrt(vr2(j))
        end do
    
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! find the velocity dispersion v_r**2
        masses = mass_total
        radius = rad
        density = rho_total
        
        
        do j = 1,nmax
    
          p = 0.0d0
          rr = radius(j)
          dr = radius(nmax) / nmax
          do jj = j,nmax
          
          m  = masses(jj)
          rh = density(jj)
          rr = rr + dr
          
          dp = rh * G * m / rr**2 * dr
          p = p + dp
        end do
      
        vr2(j) = 1/density(j) * p
        vr(j) = sqrt(vr2(j))
      enddo
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! find the accelerations felt by the particles and center of mass
        masses = mass_total
        radius = rad
        density = rho_total
    
    
        do j = 1,nmax
          rr = radius(j)
          m  = masses(j)
          acceleration(j) = G * m / rr**2
        end do
    
        acceleration_particle = acceleration
        nmerge = 50
        acc_merge = acceleration(nmerge)
        rad_merge = rad(nmerge)
        
        do j = 1, nmerge
          rr = radius(j)
          m  = masses(j)
    
    ! smoothed acceleration
          acc = G * m / (rr**2 + .1* (rad_merge -rr)) 
          acceleration_particle(j) = acc
          
        end do
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! rederive the masses from the new particle acceleration
        radius = rad
        dr = rad(2) - rad(1)
    
    ! find the accelerations felt by the particles and center of mass
        radius = rad
        
        do j = 1, nmax
          rr = radius(j)
          new_mass(j) = rr**2 * acceleration_particle(j) / G
          new_rho(j)  = new_mass(j) / (4 * pi * rr * rr * dr)
        end do
    
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! find the velocity dispersion v_r**2 using the new density and masses
        masses = new_mass
        radius = rad
        density = new_rho
        
    
        do j = 1, nmax
    
          p = 0.0d0
          rr = radius(j)
          dr = radius(nmax) / nmax
          do jj = j,nmax
          
          m  = masses(jj)
          rh = density(jj)
          rr = rr + dr
          
          dp = rh * G * m / rr**2 * dr
          p = p + dp
        end do
        
        new_vr2(j) = 1/density(j) * p
        new_vr(j) = sqrt(new_vr2(j))
        
      end do
    
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! extend the values to large rmax
    do  j= nmax+1, ntotal
      mass_total(j) = mass_total(nmax)
      mass_halo(j) = mass_halo(nmax)
      mass_disk(j) = mass_disk(nmax)
      mass_bulge(j) = mass_bulge(nmax)
      new_mass(j) = new_mass(nmax)
    
    !  rho_total(j) = 1e-3
    !  new_rho(j) = new_rho(nmax)
      rho_total(j) = 0.0d0
      new_rho(j)   = 0.0d0
    
      vr(j)      = 1d-6
      vr2(j)     = 1d-6
      new_vr(j)  = 1d-6
      new_vr2(j) = 1d-6
    
      m = mass_total(nmax)
      rr = rad(nmax) + dr*(j - nmax)
      rad(j) = rr
      acc = G * m / rr**2  
      acceleration_particle(j) = acc
      acceleration(j) = acc
    
    end do
    
    
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! normalize to the unit mass
    
    do  j= 1, ntotal
      mass_total(j)  = mass_total(j) / 7.13333d0
      mass_halo(j)   = mass_halo(j)  / 7.13333d0
      mass_disk(j)   = mass_disk(j)  / 7.13333d0
      mass_bulge(j)  = mass_bulge(j) / 7.13333d0
      new_mass(j)    = new_mass(j)   / 7.13333d0
    
      rho_total(j)   = rho_total(j)  / 7.13333d0
      new_rho(j)     = new_rho(j)    / 7.13333d0
     
      vr(j)          = vr(j)      / 7.13333d0
      vr2(j)         = vr2(j)     / 7.13333d0
      new_vr(j)      = new_vr(j)  / 7.13333d0
      new_vr2(j)     = new_vr2(j) / 7.13333d0
    
      rad(j)         = rad(j) 
    
      acceleration_particle(j) = acceleration_particle(j) / 7.13333d0
      acceleration(j)          = acceleration(j)  / 7.13333d0
    
    !!  write(11,*) rad(j), new_rho(j), new_mass(j),  new_vr(j)
    end do
    
    
    
    pscale = 1.0d0
    
    
    
    !% ! tabulate the right hand side of the dynamical friction formula
    !% xmax = 10
    !% x = 0.0d0
    !% dx = xmax / j
    !% do j = 1,j
    !%   x = x + dx
    !%   rhs(j) = erf(x) - 2.0*x / sqrt(pi) * exp(-x**2)
    !%   xx(j) = x
    !% end do
    
    
    
    
    end subroutine SIMR_INIT_DISTRIBUTION
    
    subroutine SIMR_CLEANUP()
        ! Running SPAM in this way (f2py) causes memory leaks.  
        ! This fixes that!

        deallocate(ival11)
        deallocate(ival22)
        deallocate(ivaln)
        deallocate(df_force11)
        deallocate(df_force22)
        deallocate(df_forcen)
        deallocate(c3n)

    end subroutine SIMR_CLEANUP

end module custom_runs_module
