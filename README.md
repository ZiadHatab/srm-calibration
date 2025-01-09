# Symmetric-Reciprocal-Match (SRM) Calibration

A self-calibration procedure for vector network analyzers (VNAs) that uses partial defined standards.

## **Basic principle**

SRM calibration involves measuring multiple partially defined standards. The measurements includes:

- At least 3 unique one-port measurements, such as short, open, and match.
- A two-port reciprocal device.
- One-port measurement using the two-port device and the one-port standards.
- A match standard at each port to set the reference impedance (can be non-linearly fit if model provided).

You can reduced the symmetric one-port standards to at least 2, if the two-port device is symmetric (this is off by default and is not stable).

During calibration, all standards are not specified except for the match standard. However, an estimate of the standards should be provided to resolve order/sign ambiguity during the calculations. The picture below provides a diagrammatic description of the standards. See [1] for details on an alternative implementation of the network-load standard using half of the network (also see CPW example below).

![](Images/srm_standards_definition.png)
_SRM standards. Match is not shown, but could be part of the symmetric one-port standards. There should be at least 3 different one-port standards measured_

## Code requirements

For the script [`srm.py`][srm] You only need [`numpy`](https://github.com/numpy/numpy), [`scikit-rf`](https://github.com/scikit-rf/scikit-rf) and [`scipy`](https://github.com/scipy/scipy) installed in your Python environment. To install these packages, run the following command:

```bash
python -m pip install numpy scikit-rf scipy -U
```

Regarding the dependencies for the example files, simply refer to the header of each file.

## Sample code

The [`srm.py`][srm] file should be located in the same folder as your main script, which is loaded through the import command. Switch terms not included in code below, but the script support switch terms (just read the variable definition in [`srm.py`][srm]).

```python
import skrf as rf
import numpy as np

from skrf.media import Coaxial # to create estimated values

from srm import SRM # my code

# import measurements from symmetric one-port standards as two-port networks
open_meas  = rf.Network('open_meas.s2p')
short_meas = rf.Network('short_meas.s2p')
match_meas = rf.Network('match_meas.s2p')

# import measurement of reciprocal network
reciprocal_meas = rf.Network('reciprocal_meas.s2p')

# import measurments of network-load standards from either ports
recip_open_A_meas  = rf.Network('recip_open_A_meas.s2p').s11
recip_short_A_meas = rf.Network('recip_short_A_meas.s2p').s11
recip_match_A_meas = rf.Network('recip_match_A_meas.s2p').s11

# create estimate for symmetric one-port standards and reciprocal network
freq = open_meas.frequency
coax = Coaxial(freq, Dint=1.270e-3, Dout=2.92e-3)

open_est  = coax.open()
short_est = coax.short()
match_est = coax.match()
reciprocal_est = coax.line(16e-3, 'm')

# SRM calibration
cal = SRM(symmetric=[open_meas, short_meas, match_meas], 
          est_symmetric=[open_est, short_est, match_est], # only one is required (at best open or short) 
          reciprocal=reciprocal_meas,
          est_reciprocal=reciprocal_est,
          reciprocal_GammaA=[recip_open_A_meas, recip_short_A_meas, recip_match_A_meas], 
          matchA=match_meas.s11, matchB=match_meas.s22, # ref impedance is now defined to the match standard
          # use_half_network=True  # if network-loads use half-network approach
          )
cal.run()

dut_meas  = rf.Network('dut_meas.s2p') # import dut measurement
dut_cal = cal.apply_cal(dut_meas)      # apply calibration
```

## Examples

### Coaxial measurements

This example uses standard coaxial components. It is crucial to ensure that the adapter used in the two-port measurement (middle picture below) matches the length of the adapter used to create the network-load (rightmost picture).

![](./Images/load.png) | ![](./Images/adapter.png) | ![](./Images/adapter_load.png)
:-: | :-: | :-:
_Symmetric load_ | _Reciprocal network_ | _Network-load_

Below are the results of measuring a verification kit. The ripples observed in the offset short measurements are discussed in detail in [1]. For the SOLR calibration, the SOL standards were pre-characterized by the manufacturer.

![](./Images/srm_solr_comparison.png)
_Comparison between calibrated verification kit using SRM and SOLR_

### CPW numerical simulation

This example showcases how the SRM method can be used for on-wafer applications. In the example, I provide two scenarios: one using a full-network and the other using a half-network. The latter case is illustrated below.

![](./Images/cpw_example.png)
_Illustration of CPW structures implementing the half-network approach of SRM calibration. The match standard is optional if the symmetric impedance standard is reused as the match standard._

The simulation outcome is not surprising: it yields exact results, as indicated by the error vector graph of the calibrated DUT (step-impedance), which approaches zero and is limited only by numerical precision of the software.

![](./Images/cpw_error.jpg)
_Error vector comparing full- and half-network approaches of SRM calibration_

## Crediting

If you use this method in your research or publications, please cite our main paper [1]. For use of the measurement data in your own publications, please reference dataset [2].

## References

[1] Z. Hatab, M. E. Gadringer and W. Bösch, "Symmetric-Reciprocal-Match Method for Vector Network Analyzer Calibration," in _IEEE Transactions on Instrumentation and Measurement_, vol. 73, pp. 1-11, 2024, Art no. 1001911, doi: [10.1109/TIM.2024.3350124](https://doi.org/10.1109/TIM.2024.3350124), e-print: <https://arxiv.org/abs/2309.02886>

[2] Z. Hatab, "Symmetric-Reciprocal-Match Calibration: Dataset and Code". Graz University of Technology, Sep. 07, 2023. doi: [10.3217/z5zhh-5qh04](https://doi.org/10.3217/z5zhh-5qh04).

## License

Feel free to do whatever you want with the code under the terms of the [BSD-3-Clause license](https://github.com/ZiadHatab/srm-calibration/blob/main/LICENSE).

[srm]: https://github.com/ZiadHatab/srm-calibration/blob/main/srm.py
