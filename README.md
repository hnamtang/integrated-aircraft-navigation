# Integrated Aircraft Navigation

## Project - GPS/INS Integration

<u>Due</u>: via [email](mailto:maarten.uijtdehaag@tu-berlin.de)

This is the repository for the project Integrated Aircraft Navigation at TU Berlin in summer semester 2024

## Table of Contents

1. [Tasks](#tasks)
2. [To be Submitted](#tobesubmitted)
3. [Notes](#notes)
4. [Data Files](#datafiles)
5. [Provided Python Functions](#providedpythonfunctions)
6. [References](#references)

<a id="tasks"></a>

## Tasks

- [ ] **Loose integration** of GPS and INS to obtain an estimate of the aircraft position $(L, \lambda, h)$
- [ ] **Tight integration** of GPS and INS to obtain an estimate of the aircraft position $(L, \lambda, h)$
- [ ] **Detector** of possible pseudorange **step errors** in one of the pseudoranges.

<a id="tobesubmitted"></a>

## To be Submitted

- [ ] A written report in digital form (Word or PDF).
- [ ] Clearly state your assumptions, reference your sources, and outline your methodology.
- [ ] Derive which algorithms you have used to calculate the 3D positions and error detection.
- The plots should include at least:
  - [ ] a latitude versus longitude plot that includes INS, GPS, and Integrated solutions in different colors,
  - [ ] a height versus elapsed time plot,
  - [ ] number of available GPS satellites as a function of time,
  - [ ] an error plot showing the difference between the GPS and GPS/INS solution in East, North, Up directions (you may include the covariance in this plot),
  - [ ] a plot showing the filter innovations,
  - [ ] and other plots you deem important to analyze the performance of your filter.
- [ ] Include your <u>properly commented well-structured</u> Python code for the loose coupling and for the tight integration.

<a id="notes"></a>

## Notes

In your program you should run through all the times in `t_gps`.
For each of these time tags there will be an INS position available.
In terms of GPS, there exist multiple measurements (i.e., pseudoranges) for each time tag (remember you need 4 or more ranges to compute a position).
Therefore, for each time tag you must

1. extract the pseudoranges,
2. calculate the satellite and user position, and
3. integrate these with the INS data

<a id="datafiles"></a>

## Data Files

The data was collected using a DC-3 equipped with a NovAtel OEM-4-dual-frequency receiver (L1 and L2) and a tactical-grade inertial measurement unit (IMU) [[1]](#1).
The latter has

- a gyro drift of about $1^{\circ}/\mathrm{hr}$,
- an accelerometer bias of about $1\;\mathrm{mg}$,
- a gyro angle random walk (ARW) of about $0.5^{\circ}/\sqrt{\mathrm{hr}}$, and
- an accelerometer velocity random walk (VRW) of $0.2\;\mathrm{m/s/}\sqrt{\mathrm{hr}}$.

<u>INS files contain the following data (NO raw data)</u>:

- `t_ins`: an $M\times 1$ vector containing the time tags for the M positions and attitudes output by the inertial system.
- `t_llh_ins': an $3\times M$ matrix whose columns contain the LLH position computed by the INS at each time epoch \[rad, rad, m\].
- `rpy_ins`: not relevant for this project.

<u>GPS files contain the following data</u>:

- `t_gps`: an $N\times 1$ vector containing the time tags for the N raw measurements (pseudorange and carrier-phase) output by the GPS receiver.
- `svid_gps`: an $N\times 1$ vector containing the satellite identification (SVID) number for the corresponding measurement.
- `pr_gps`: an $N\times 1$ vector containing the pseudorange measurement \[m\].
- `adr_gps`: an $N\times 1$ vector containing carrier phase measurement \[m\].
- `ephem`: an $22\times 28$ matrix containing the orbital parameters (ephemerides) that are required to compute the satellite position at a specific time.

There are 3 different GPS data files that all must be processed and analyzed. The templates already include code that extracts the data from these files.

<a id="providedpythonfunctions"></a>

## Provided Python Functions

Two functions are given within the `gnss_routines.pyc` files to

1. extract the satellite position,
2. correct the pseudoranges for the satellite clock offset, and
3. compute the GPS position.

The first function is:

```python
(sv_ecef, sv_clock) = compute_svpos_svclock_compensate(gpstime, svid, pr, ephem)
```

This function takes as inputs the current GPS time (`gpstime`), an array with all the satellite IDs for the current time (`svid`), and array with all the pseudoranges for the current time (`pr`) and the `ephem`. It returns two new arrays for the satellite positions in ECEF (`sv_ecef`) and the satellite clock offsets from GPS time (`sv_clock`). Output `pr_new`is the pseudorange corrected for the satellite clock offset.

The second function is:

```python
(r_ecef_gps, user_clock) = compute_pos_ecef(gpstime, pr, sv_ecef, sv_clock)
```

This function takes as inputs the array of pseudoranges, the corresponding satellite positions in ECEF, and the satellite clock errors. It returns the user position in ECEF through `r_ecef_gps`and the user clock error (in meters) through `sv_clock`.

<a id="references"></a>

## References

<a id="1">[1]</a>
Campbell, J. L., M. Uijt de Haag, F. van Graas, T. Arthur, T. J. Dickman, DC-3 Flying Laboratory - Navigation Sensor and Remote Sensing Flight Test Results, ION GNSS _17<sup>th</sup> International Technical Meeting of the Satellite Division_, Sept. 2004.
