program armsgb_driver_allsy
  use RTM_General
  implicit none

  integer, parameter :: nin = 11, nout = 12, nmax = 500, ncmax = 128
  real(RP), parameter :: thr_cld = 1.0e-8_RP
  character(len=256) :: fin, fout, magic, arg1, arg2
  character(len=32)  :: sensor_id
  character(len=256) :: coef_path
  integer :: nlev, nchan, i, j, ilevel, want_jac, ios, nargs, ncol, ncld, ic, ccnt
  integer :: cflag(4)
  real(RP) :: p(nmax), t(nmax), q(nmax), o3(nmax), freq(ncmax)
  real(RP) :: zg(nmax), clwc(nmax), crwc(nmax), cswc(nmax), ciwc(nmax)
  real(RP) :: zenith_deg, psfc, t2m, q2m
  real(RP) :: dz, c1, c2, c3, c4, tm, lambda, mu, r_e
  real(RP), allocatable :: tb(:), K(:,:)
  logical :: valid_ok

  type(RTMCoef)       :: coeff
  type(InputProfile)  :: user_prof, user_prof_ad
  type(OutputProfile) :: rts_prof, rts_prof_ad

  nargs = command_argument_count()
  if (nargs >= 1) then
    call get_command_argument(1, arg1)
    if (trim(arg1) == '--channels') then
      sensor_id = 'MP_2500A_V2'
      coef_path = './fix/'
      call Load_SensorCoef(coeff, sensor_id, coef_path)
      fout = 'channels.txt'
      if (nargs >= 2) call get_command_argument(2, fout)
      open(nout, file=trim(fout), status='replace', action='write')
      write(nout,'(i0)') coeff%N_Channels
      write(nout,'(*(es24.16))') (coeff%Frequency(i), i=1,coeff%N_Channels)
      close(nout)
      call Unload_SensorCoef(coeff)
      stop
    end if
  end if

  if (nargs < 2) then
    write(*,*) 'Usage: armsgb_driver_allsy profile.in tb.out'
    stop 2
  end if
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)

  zg = 0.0_RP; clwc = 0.0_RP; crwc = 0.0_RP; cswc = 0.0_RP; ciwc = 0.0_RP

  open(nin, file=trim(fin), status='old', action='read')
  read(nin,*) magic
  read(nin,*) nlev
  if (nlev > nmax) stop 'nlev too large'
  read(nin,*) (p(i), i=1,nlev)
  read(nin,*) (t(i), i=1,nlev)
  read(nin,*) (q(i), i=1,nlev)
  read(nin,*) nchan
  if (nchan > ncmax) stop 'nchan too large'
  read(nin,*) (freq(i), i=1,nchan)
  read(nin,*) zenith_deg
  read(nin,*) psfc, t2m, q2m
  read(nin,*) want_jac
  read(nin,*,iostat=ios) (o3(i), i=1,nlev)
  if (ios /= 0) o3(1:nlev) = 1.0e-7_RP
  read(nin,*,iostat=ios) (zg(i), i=1,nlev)
  if (ios == 0) then
    read(nin,*,iostat=ios) (clwc(i), i=1,nlev)
    read(nin,*,iostat=ios) (crwc(i), i=1,nlev)
    read(nin,*,iostat=ios) (cswc(i), i=1,nlev)
    read(nin,*,iostat=ios) (ciwc(i), i=1,nlev)
  end if
  close(nin)
  where (clwc < 0.0_RP) clwc = 0.0_RP
  where (crwc < 0.0_RP) crwc = 0.0_RP
  where (cswc < 0.0_RP) cswc = 0.0_RP
  where (ciwc < 0.0_RP) ciwc = 0.0_RP

  ! --- cloud presence flags, packed exactly like K_Batch_NC ---
  cflag = 0
  if (maxval(clwc(1:nlev)) > thr_cld) cflag(1) = 1
  if (maxval(crwc(1:nlev)) > thr_cld) cflag(2) = 1
  if (maxval(cswc(1:nlev)) > thr_cld) cflag(3) = 1
  if (maxval(ciwc(1:nlev)) > thr_cld) cflag(4) = 1
  ncld = sum(cflag)

  sensor_id = 'MP_2500A_V2'
  coef_path = './fix/'
  call Load_SensorCoef(coeff, sensor_id, coef_path)
  if (nchan /= coeff%N_Channels) then
    write(*,*) 'ERROR: input nchan=', nchan, ' ARMS nchan=', coeff%N_Channels
    stop 3
  end if

  call UserProfile_Init(user_prof, nlev, ncld)
  user_prof%Level_Pressure    = p(1:nlev)
  user_prof%Level_Temperature = t(1:nlev)
  user_prof%Level_H2O         = q(1:nlev)
  user_prof%Level_O3          = o3(1:nlev)
  user_prof%Surface_Pressure  = psfc
  user_prof%Temperature_2m    = t2m
  user_prof%H2O_2m            = q2m
  user_prof%Sensor_ZenithAngle= zenith_deg
  user_prof%GasAbsorp_Type    = LBLRegressV1

  if (ncld > 0) then
    ccnt = 0
    if (cflag(1) == 1) then
      ccnt = ccnt + 1
      user_prof%Cloud(ccnt)%Type = WATER_CLOUD
      do ilevel = 1, nlev-1
        dz = zg(ilevel) - zg(ilevel+1); c1 = 0.5_RP*(clwc(ilevel)+clwc(ilevel+1)); tm = 0.5_RP*(t(ilevel)+t(ilevel+1))
        if (c1 > thr_cld) then
          user_prof%Cloud(ccnt)%Water_Content(ilevel) = c1*dz
          if (tm >= 265.0_RP) then
            user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = max(4.0_RP, min(30.0_RP, 10.0_RP*(c1*1e3)**0.33_RP))
          else
            user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = 4.0_RP
          end if
        else
          user_prof%Cloud(ccnt)%Water_Content(ilevel) = 0.0_RP
          user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = 0.0_RP
        end if
      end do
    end if
    if (cflag(2) == 1) then
      ccnt = ccnt + 1
      user_prof%Cloud(ccnt)%Type = RAIN_CLOUD
      do ilevel = 1, nlev-1
        dz = zg(ilevel) - zg(ilevel+1); c2 = 0.5_RP*(crwc(ilevel)+crwc(ilevel+1)); tm = 0.5_RP*(t(ilevel)+t(ilevel+1))
        if (c2 > thr_cld) then
          user_prof%Cloud(ccnt)%Water_Content(ilevel) = c2*dz
          if (c2 > 1.0e-6_RP) then
            user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = max(50.0_RP, min(200.0_RP, 14.0_RP*(c2*1e3)**0.21_RP))
          else
            if (tm >= 273.15_RP) then
              user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = 52.0_RP + 0.4_RP*(tm-273.15_RP)
            else
              user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = 50.0_RP
            end if
          end if
        else
          user_prof%Cloud(ccnt)%Water_Content(ilevel) = 0.0_RP
          user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = 0.0_RP
        end if
      end do
    end if
    if (cflag(3) == 1) then
      ccnt = ccnt + 1
      user_prof%Cloud(ccnt)%Type = SNOW_CLOUD
      do ilevel = 1, nlev-1
        dz = zg(ilevel) - zg(ilevel+1); c3 = 0.5_RP*(cswc(ilevel)+cswc(ilevel+1)); tm = 0.5_RP*(t(ilevel)+t(ilevel+1))
        if (c3 > thr_cld) then
          user_prof%Cloud(ccnt)%Water_Content(ilevel) = c3*dz
          if (c3 > 1.0e-6_RP) then
            user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = max(50.0_RP, min(500.0_RP, 200.0_RP*(c3*1e3)**0.18_RP))
          else
            if (tm < 273.15_RP) then
              user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = max(50.0_RP, min(300.0_RP, 80.0_RP+1.2_RP*(273.15_RP-tm)))
            else
              user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = 50.0_RP
            end if
          end if
        else
          user_prof%Cloud(ccnt)%Water_Content(ilevel) = 0.0_RP
          user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = 0.0_RP
        end if
      end do
    end if
    if (cflag(4) == 1) then
      ccnt = ccnt + 1
      user_prof%Cloud(ccnt)%Type = ICE_CLOUD
      do ilevel = 1, nlev-1
        dz = zg(ilevel) - zg(ilevel+1); c4 = 0.5_RP*(ciwc(ilevel)+ciwc(ilevel+1)); tm = 0.5_RP*(t(ilevel)+t(ilevel+1))
        if (c4 > thr_cld) then
          user_prof%Cloud(ccnt)%Water_Content(ilevel) = c4*dz
          if (tm <= 273.15_RP) then
            lambda = 24.0e-6_RP*exp(-0.049_RP*(tm-273.15_RP))
          else
            lambda = 6.8e-6_RP*exp(-0.096_RP*(tm-273.15_RP))
          end if
          mu = 0.13_RP*(lambda*1.0e6_RP)**0.64_RP - 2.0_RP
          r_e = (159.0_RP*GAMMA_FUNC(3.2_RP+mu))/(GAMMA_FUNC(2.8_RP+mu)*(lambda*1.0e6_RP)**0.41_RP)
          user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = max(5.0_RP, min(150.0_RP, r_e))
        else
          user_prof%Cloud(ccnt)%Water_Content(ilevel) = 0.0_RP
          user_prof%Cloud(ccnt)%Effective_Radius(ilevel) = 0.0_RP
        end if
      end do
    end if
  end if

  call Check_UserProfile(user_prof, valid_ok)
  if (.not. valid_ok) then
    write(*,*) 'ERROR: invalid ARMS input profile'
    stop 4
  end if

  allocate(tb(nchan))
  call RTSProfile_Init(rts_prof, nlev, nchan)
  call RTM_Forward(user_prof, rts_prof, coeff)
  tb(1:nchan) = rts_prof%Brightness_Temperature(1:nchan)

  ! --- all-sky Jacobian via the ADJOINT, one channel at a time (as K_Batch_NC) ---
  if (want_jac == 1) then
    ncol = 2*nlev
    allocate(K(nchan, ncol)); K = 0.0_RP
    do ic = 1, nchan
      call UserProfile_Init(user_prof_ad, nlev, ncld)
      call RTSProfile_Init(rts_prof_ad, nlev, nchan)
      call zero_ad(user_prof_ad)
      rts_prof_ad%Brightness_Temperature(ic) = 1.0_RP
      call RTM_Adjoint(user_prof, rts_prof, coeff, user_prof_ad, rts_prof_ad, ChannelID=ic)
      K(ic, 1:nlev)        = user_prof_ad%Level_Temperature(1:nlev)
      K(ic, nlev+1:2*nlev) = user_prof_ad%Level_H2O(1:nlev)
      call UserProfile_Free(user_prof_ad)
      call RTSProfile_Free(rts_prof_ad)
    end do
  end if

  open(nout, file=trim(fout), status='replace', action='write')
  write(nout,'(i0)') nchan
  write(nout,'(*(es24.16))') (tb(i), i=1,nchan)
  if (want_jac == 1) then
    write(nout,'(i0,1x,i0)') nchan, ncol
    do i = 1, nchan
      write(nout,'(*(es24.16))') (K(i,j), j=1,ncol)
    end do
  end if
  close(nout)

  call UserProfile_Free(user_prof)
  call RTSProfile_Free(rts_prof)
  if (allocated(tb)) deallocate(tb)
  if (allocated(K))  deallocate(K)
  call Unload_SensorCoef(coeff)

contains

  subroutine zero_ad(P)
    type(InputProfile), intent(inout) :: P
    integer :: c
    P%Level_Temperature = 0.0_RP
    P%Level_H2O         = 0.0_RP
    P%Level_O3          = 0.0_RP
    P%Temperature_2m    = 0.0_RP
    P%H2O_2m            = 0.0_RP
    P%Surface_Pressure  = 0.0_RP
    if (P%n_Clouds > 0) then
      do c = 1, P%n_Clouds
        P%Cloud(c)%Water_Content    = 0.0_RP
        P%Cloud(c)%Effective_Radius = 0.0_RP
      end do
    end if
  end subroutine zero_ad

  FUNCTION GAMMA_FUNC(x) RESULT(g)
    REAL(RP), INTENT(IN) :: x
    REAL(RP) :: g
    REAL(RP), PARAMETER :: pi = 3.141592653589793_RP
    REAL(RP) :: z, frac
    INTEGER :: kk
    REAL(RP), DIMENSION(7) :: c = [ 0.99999999999980993_RP, 676.5203681218851_RP, -1259.1392167224028_RP, &
                                    771.32342877765313_RP, -176.61502916214059_RP, 12.507343278686905_RP, &
                                    -0.13857109526572012_RP ]
    z = x - 1.0_RP
    frac = c(1)
    DO kk = 2, 7
      frac = frac + c(kk) / (z + REAL(kk-1, RP))
    END DO
    g = SQRT(2.0_RP * pi) * (z + 5.5_RP)**(z + 0.5_RP) * EXP(-(z + 5.5_RP)) * frac
  END FUNCTION GAMMA_FUNC

end program armsgb_driver_allsy
