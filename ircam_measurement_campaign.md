# IRCAM Measurement Campaign

This document describes the measurement campaign conducted at IRCAM to evaluate the Real2Sim framework on real room impulse responses (RIRs).

The measurements were performed in collaboration with Olivier Warusfel in a room with a geometry close to a shoebox shape: the artist lodge at IRCAM.

---

## Motivation

The goal of this measurement campaign was to evaluate the proposed framework on real data.

Our framework requires acoustic devices with known directivity patterns that can be incorporated into the simulation pipeline. For this reason, we used:

- a Genelec 8030B loudspeaker, whose directivity pattern has been measured in previous work;
- a compact 32-channel Eigenmike32 microphone array, whose directivity pattern is included in the DIRPAT database used by `pyroomacoustics`.

The measured RIRs were then compared with simulated configurations reconstructed from room and device-position annotations.

---

## Measurement setup

For practical reasons, both the loudspeaker and the microphone array were placed on a rolling trolley. Since our framework is agnostic to the orientation of the microphone array, we only ensured that the loudspeaker was facing the array.

The rolling trolley could then be moved to different positions in the room. At each position, an RIR measurement was performed while maintaining a minimum distance of 1.50 m between the devices and the room boundaries.

For each measurement position, we annotated the distances from the estimated center of the devices to the room walls using a laser pointer. The annotation accuracy was approximately ±5 cm. These annotations allowed us to re-simulate the different measurement configurations.

The RIRs were measured using eight repeated sine sweeps ranging from 50 Hz to 20 kHz. The recorded signals were then deconvolved to compute the RIRs.

---

## Room description

```text
City: Paris
Country: France
Venue: IRCAM - Loges Espro
Room type: Rectangular parallelepiped
Dimensions: 9.53 × 4.72 × 2.39 m up to 4.51 m from the east wall [absorbing ceiling]
Height: 2.78 m from 4.51 m onward [reflective ceiling]
Room condition: empty
```

The room has a mostly shoebox-like geometry, but the ceiling height and ceiling material change along the room. Up to 4.51 m from the east wall, the room height is 2.39 m and the ceiling is absorbing. From this point onward, the height is 2.78 m and the ceiling is reflective.

---

## Microphone description

```text
Brand - Model: MH Acoustics EM32 Eigenmike, IRCAM
ADC used: channels 129-161 via MADI3 of the Madiface XT audio interface
Gain set in the EM32 plugin: +30 dB

Microphone height: 1.71 m above floor level for measurements 1 to 6
Microphone height: 1.35 m above floor level from measurement 7 onward
```

The microphone array used during the campaign was a 32-channel Eigenmike32. Its directivity pattern is available in the DIRPAT database and can be used in the simulation pipeline.

---

## Loudspeaker description

```text
Brand - Model: Genelec 8030 active loudspeaker
Serial no.: 8030APM6028955
Gain on the loudspeaker: Max
Crossover settings - Band tilts: default, all DIP switches off
Output channel on Madiface XT: 193 -> Main 1

Loudspeaker height above floor level: 1.71 m, measured at the physical center of the loudspeaker, up to measurement 6
Loudspeaker height above floor level: 1.35 m from measurement 7 onward
```

The directivity pattern of the Genelec 8030 loudspeaker was taken from:

> Gallien, A., Prawda, K., & Schlecht, S. J. (2024).  
> *Matching early reflections of simulated and measured RIRs by applying sound-source directivity filters.*  
> Proceedings of the AES 2024 International Acoustics & Sound Reinforcement Conference, Le Mans, France.

---

## Acquisition details

```text
Clock: Madiface XT master
High-pass filter: Max/MSP high-pass filter at 85 Hz
Purpose of the high-pass filter: avoid loudspeaker distortion
```

The high-pass filter was applied during playback to avoid distortion of the loudspeaker at low frequencies.

---

## Session description

```text
People involved: Jean-Daniel Pascal, Olivier Warusfel
Date: March 19, 2025, from 5:00 p.m. to 7:00 p.m.
Project: collaboration with CEREMA
```

The RIRs were recorded in real-life conditions. The loudspeaker and microphone array were placed on a wheeled platform measuring 2 m × 0.5 m, with a platform height of 33 cm.

The relative distance between the front face of the microphone and the front face of the loudspeaker was 1.53 m.

The positions of the microphone and loudspeaker in the room were progressively recorded throughout the session.

---

## Dataset

You can find the dataset as .mat files for 10 differents positions of the devices in the room in the following zenodo :
 [zenodo file]

---

## Calibration of the physical time origin

In the measured RIRs, the physical time origin, denoted as `t = 0`, is unknown. To calibrate the system, we fitted the direct-path time of arrival using simulation.

After this calibration step, the Real2Sim framework can be applied to the evaluation dataset.

---

## Image-source alignment

The set of image sources estimated by the framework is not directly aligned with the ground-truth image-source locations obtained from the annotations.

This misalignment is mainly due to the lack of annotation of the microphone-array orientation during the measurement campaign.

We propose to estimate this orientation a posteriori. If the correct orientation is recovered, the remaining error corresponds to a rotation of the world around the microphone array. Therefore, the room geometry can still be reconstructed in the microphone-array reference frame.

To align the estimated image-source cloud with the ground truth, we rely on the fact that the real source position is accurately estimated. The alignment procedure is as follows:

1. Superimpose the estimated source and the true source.
2. Rotate the estimated image-source cloud around the source–microphone-array axis.
3. Use an adapted Hungarian algorithm to find the best partial matching between the estimated image sources and the expected image-source locations.
4. Select the rotation that gives the best matching score.

Although the matching thresholds are set to 1 m and 20°, the image-source estimation metrics show that the final errors are well below these thresholds. These relatively large thresholds are only used to ensure that the correct matching is found, even in the presence of measurement errors or missing annotations of the room dimensions.

---

## Reflections on the rolling trolley

We used a tolerant criterion for the detection of reflections on the rolling trolley, since we did not initially expect such reflections to appear in the results.

Reflections on the trolley and on the floor were counted as bonus detections when they were correctly localized across all reflection orders.

---

## Related scripts

The measurement-based dataset can be generated using:

```bash
python RIR_generation/generate_rirs_measurement_IRCAM.py
```

The image-source matching and metric computation procedure is provided in:

```text
image_source_localization/matching_and_metrics.py
```

The image-source localization step relies on:

```text
https://github.com/Sprunckt/acoustic-sfw.git
```
