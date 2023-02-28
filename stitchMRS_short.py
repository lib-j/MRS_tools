"""
File name: stitchMRS_short.py
Script to stitch together the MRS segments of arbetary length
The 1SHORT MRS segment is assumed to be the default and the segement with the best flux calibration.
The spectra are assumed to be in ascii format.
Author: Olivia Jones & Alan Stokes (UK ATC)
Date created: 6/7/2022
Date last modified: 8/7/2022
"""

# Import packages 
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.io import ascii
from astropy.nddata import StdDevUncertainty

# To deal with 1D spectrum
from specutils import Spectrum1D
from specutils.spectra import SpectralRegion
from specutils import SpectrumList
from specutils.manipulation import extract_region, noise_region_uncertainty
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler


class MRSstitch:

    def __init__(self, spectral_files):
        # Private variable due to underscore only accessible by this class instance.
        self._spectral_files = spectral_files
        self._spec_seg_permanent_store = None

    def mrs_spectral_stitch(self):
        """ stitches together MRS files into 1 spectral.
        @:parameter spectral_files: list of ascii MRS spectral files that need stitiching together.
        @:return 1DSpectrum object with the stitiched MRS segemts mergeded together.
        """

        spec_list, truth_index = MRSstitch._create_spec_collection_find_shortest_wav_mrs_seg(self._spectral_files)

        # This is to store the original spectral data to check stitching as original is deleted during stitching
        self._spec_seg_permanent_store = spec_list.copy()

        # extract MRS segment with the shortest wavelength from spectral list
        first_mrs_segment = spec_list[truth_index]
        del spec_list[truth_index]

        final_stiched_spec = MRSstitch._do_stitching(spec_list, first_mrs_segment)

        print("The MRS spectra have now been stitched together!")

        return final_stiched_spec

    def plot_spectra(self, stichedspectra):
        """ Plot the spectrum & each MRS spectrum to inspect stitching.
        @:param stichedspectra: The final MRS spectra - of each segment stitiched together.
        @:type stichedspectra: Spectrum1D
        @:return: None
        """
        plt.figure(figsize=(6, 4))
        plt.plot(stichedspectra.spectral_axis, stichedspectra.flux)
        for spectra in self._spec_seg_permanent_store:
            plt.plot(spectra.spectral_axis, spectra.flux)
        plt.xlabel('Wavelength (microns)')
        # noinspection PyStringFormat
        plt.ylabel("Flux ({:latex})".format(stichedspectra.flux.unit))
        plt.tight_layout()
        plt.show()
        plt.close()

    @staticmethod
    def _plot_overlap_mrs_spec_region(short_mrs_overlap_seg, long_mrs_overlap_seg):
        """
        Make a plot to check the MRS spectral wavelength overlap region.
        @:param short_mrs_overlap_seg: short wavelength overlap region.
        @:param long_mrs_overlap_seg:  long wavelength overlap region.
        @:return: None
        """
        plt.figure(figsize=(6, 4))
        plt.plot(short_mrs_overlap_seg.spectral_axis, short_mrs_overlap_seg.flux)
        plt.plot(long_mrs_overlap_seg.spectral_axis, long_mrs_overlap_seg.flux)
        plt.xlabel('Wavelength (microns)')
        # noinspection PyStringFormat
        plt.ylabel("Flux ({:latex})".format(short_mrs_overlap_seg.flux.unit))
        plt.tight_layout()
        plt.show()
        plt.close()

    @staticmethod
    def _create_spec_collection_find_shortest_wav_mrs_seg(spectral_files):
        """
        Fuunction to put all spectra segments in the ascii files into a SpectrumList and
        find the index of the segment with the shortest wavelenghts.
        It is better coding practice to break up long codes into methods!
        The underscore makes the function private - it will not show in autocomplete methonds for the class in an IDE.
        @:param spectral_files: list of ascii MRS spectral files that need stitiching together.
        @:return:  tuple containing the read in spec list and the index of the shortest mrs segment.
        @:rtype: tuple(SpectrumList, integer)
        """
        # Put all the 12 MIRI MRS segments into a spectralList
        spec_list = SpectrumList()

        # Used to find the shortest wavelength MRS segment as assume this is the best flux calibrated
        truth_index = 0
        lastSeen = sys.maxsize

        # Populate spectralList with the MIRI MRS segments
        for specfile in spectral_files:
            data = ascii.read(specfile)
            wavelength = data['col1'] * u.micron
            flux = data['col2'] * u.Jy
            # uncertainty = data['col3']
            # spec1d = Spectrum1D(spectral_axis=wavelength, flux=flux, uncertainty=StdDevUncertainty(uncertainty))
            spec1d = Spectrum1D(spectral_axis=wavelength, flux=flux)
            # spec_list.append(spec1d)

            # Alternatively estimate the noise from each of the MRS spectral segment
            # NB: This is not ideal as it includes lines and broadband features.
            noise_region = SpectralRegion(wavelength[0], wavelength[-1])
            spec1d_w_unc = noise_region_uncertainty(spec1d, noise_region)
            spec_list.append(spec1d_w_unc)

            # update truth index based off the currently seen the smallest wavelength.
            if wavelength[0].value < lastSeen:
                lastSeen = wavelength[0].value
                truth_index = len(spec_list) - 1

        return spec_list, truth_index

    @staticmethod
    def _findNextShortestAndRemove(spec_list):
        # find next shortest MRS segment in the spectral list
        next_shorest_wav_segment = sys.maxsize
        next_shortest_index = 0
        for index, mrs_spectral_segment in enumerate(spec_list):
            if mrs_spectral_segment.spectral_axis[0].value < next_shorest_wav_segment:
                next_shorest_wav_segment = mrs_spectral_segment.spectral_axis[0].value
                next_shortest_index = index

        # remove next shortest MRS segment from list
        next_shortest = spec_list[next_shortest_index]
        del spec_list[next_shortest_index]
        return spec_list, next_shortest

    @staticmethod
    def _find_overlap(shortest_mrs_segment, next_shortest):
        # Find wavelength overlap values
        max_overlap_wav = shortest_mrs_segment.spectral_axis[-1]
        min_overlap_wav = next_shortest.spectral_axis[0]

        # Find and extract the places where the spectral segments overlap
        # TODO: This is slow - find alternative method
        short_region = SpectralRegion(max_overlap_wav, min_overlap_wav)
        long_region = SpectralRegion(min_overlap_wav, max_overlap_wav)
        short_overlap_seg = extract_region(shortest_mrs_segment, short_region)
        long_overlap_seg = extract_region(next_shortest, long_region)
        return short_overlap_seg, long_overlap_seg, min_overlap_wav, max_overlap_wav

    @staticmethod
    def _generate_scaled_mrs_segment(next_shortest, flux_scale):
        # Correct the flux [and uncertainty???] in the longer wavelength MRS segment
        # TODO: Check if this is correct way to handel the uncertainty
        scaled_flux = next_shortest.flux * flux_scale
        # scaled_uncertainty = next_shortest.uncertainty.array * flux_scale * next_shortest.flux.unit
        scaled_uncertainty = next_shortest.uncertainty

        scaled_spectrum = Spectrum1D(spectral_axis=next_shortest.spectral_axis, flux=scaled_flux,
                                     uncertainty=StdDevUncertainty(scaled_uncertainty))
        return scaled_spectrum

    @staticmethod
    def _correct_flux_in_overlap_region(
            scaled_spectrum,
            short_overlap_seg,
            next_shortest,
            shortest_mrs_segment,
            min_overlap_wav,
            max_overlap_wav):
        # In region where spectral overlap regrid to same wavelength and determine average flux
        # 1) Put on same wavelength scale
        fluxcon = FluxConservingResampler()
        new_spec_fluxcon = fluxcon(scaled_spectrum, short_overlap_seg.spectral_axis)

        # 2) Determine the mean flux in the overlap region
        meanflux = np.mean([new_spec_fluxcon.flux, short_overlap_seg.flux], axis=0) * new_spec_fluxcon.flux.unit

        # 3) TODO: Determine the max uncertainty in overlap region
        max_error_array = []
        for (array1, array2) in zip(short_overlap_seg.uncertainty.array, new_spec_fluxcon.uncertainty.array):
            if array1 > array2:
                max_error_array.append(array1)
            else:
                max_error_array.append(array2)

        updated_uncertainty = StdDevUncertainty(max_error_array * next_shortest.flux.unit)

        # 4) Make a Spectrum1D of the overlap region
        overlap_mrs_segment = Spectrum1D(
            spectral_axis=new_spec_fluxcon.spectral_axis, flux=meanflux, uncertainty=updated_uncertainty)

        # 5) Combine the 3 segments shorter wav, modified overlap wav region, longer wav into one Spectrum1D object
        # TODO: think I am one array index out on either side. Fix!
        shortest_wav_region = SpectralRegion(shortest_mrs_segment.spectral_axis[0], min_overlap_wav)
        longer_wav_region = SpectralRegion(max_overlap_wav, scaled_spectrum.spectral_axis[-1])
        overlap_wav_region = SpectralRegion(overlap_mrs_segment.spectral_axis[1],
                                            overlap_mrs_segment.spectral_axis[-2])
        return shortest_wav_region, longer_wav_region, overlap_wav_region, overlap_mrs_segment

    @staticmethod
    def _find_flux_scale(short_overlap_seg, long_overlap_seg):
        # Use the overlap spectral region to determine the flux scale factor
        flux_scale = short_overlap_seg.mean() / long_overlap_seg.mean()

        # There is a line at ~15.5 microns which can mess up flux calibration - make sure this is not used.
        # For the MRS band join which can be affected by the ~15.5 [NeIII] line exclude the line in flux correction.
        # TODO: Make generic to mask any line with S/N > threshold_value
        # Line wavelengths 15.51 - 15.58
        if (short_overlap_seg.spectral_axis.value.min() < 15.51) and (
                short_overlap_seg.spectral_axis.value.max() > 15.51):
            # Set spectral regions where the line maybe present
            exclude_regions_short = SpectralRegion([(short_overlap_seg.spectral_axis.min(), 15.51 * u.um)])
            exclude_regions_long = SpectralRegion([(long_overlap_seg.spectral_axis.min(), 15.51 * u.um)])

            # Extract the overlap regions to exclude the line
            short_overlap_seg_noline = extract_region(short_overlap_seg, exclude_regions_short)
            long_overlap_seg_noline = extract_region(long_overlap_seg, exclude_regions_long)

            # Determine the scale factor when no line is present
            flux_scale = short_overlap_seg_noline.mean() / long_overlap_seg_noline.mean()
            # MRSstitch.plot_overlap_mrs_spec_region(short_overlap_seg_noline, long_overlap_seg_noline)
        print("The scale factor is:", flux_scale)
        return flux_scale

    @staticmethod
    def _stitch_two_regions_together(
            shortest_mrs_segment, shortest_wav_region, scaled_spectrum,
            longer_wav_region, overlap_mrs_segment, overlap_wav_region):
        shortspec = extract_region(shortest_mrs_segment, shortest_wav_region)
        longspec = extract_region(scaled_spectrum, longer_wav_region)
        overlapspec = extract_region(overlap_mrs_segment, overlap_wav_region)

        # Make a new spectral axis
        # TODO: Check no duplicate wavelengths
        new_spectral_axis = np.concatenate([shortspec.spectral_axis.value,
                                            overlapspec.spectral_axis.value,
                                            longspec.spectral_axis.value]) * shortspec.spectral_axis.unit

        resampler = LinearInterpolatedResampler(extrapolation_treatment='zero_fill')

        # Put on the new combined spectral axis
        new_spec1 = resampler(shortspec, new_spectral_axis)
        new_spec2 = resampler(overlapspec, new_spectral_axis)
        new_spec3 = resampler(longspec, new_spectral_axis)

        # Combine the 3 spectral parts together to make new shortest mrs segment.
        return new_spec1 + new_spec2 + new_spec3

    @staticmethod
    def _do_stitching(spec_list, shortest_mrs_segment):
        """ does the actual stitching.

        @:param spec_list: the list of spectral MRS segments.
        @:param shortest_mrs_segment: the shortest segment.
        @:return: the stiched mrs segments.
        """
        # keep removing arrays as processed each MRS segment
        while len(spec_list) > 0:
            spec_list, next_shortest = MRSstitch._findNextShortestAndRemove(spec_list)

            short_overlap_seg, long_overlap_seg, min_overlap_wav, max_overlap_wav = (
                MRSstitch._find_overlap(
                    shortest_mrs_segment,
                    next_shortest))

            flux_scale = MRSstitch._find_flux_scale(
                short_overlap_seg,
                long_overlap_seg)

            scaled_spectrum = MRSstitch._generate_scaled_mrs_segment(
                next_shortest,
                flux_scale)

            shortest_wav_region, longer_wav_region, overlap_wav_region, overlap_mrs_segment = (
                MRSstitch._correct_flux_in_overlap_region(
                    scaled_spectrum,
                    short_overlap_seg,
                    next_shortest,
                    shortest_mrs_segment,
                    min_overlap_wav,
                    max_overlap_wav))

            shortest_mrs_segment = MRSstitch._stitch_two_regions_together(
                shortest_mrs_segment,
                shortest_wav_region,
                scaled_spectrum,
                longer_wav_region,
                overlap_mrs_segment,
                overlap_wav_region)

        return shortest_mrs_segment


if __name__ == "__main__":
    # Load the data
    datapath = "/Users/ojones/Data/SMPLMG58/5July_reduction/extracted_spec"
    # extracted1dspecFiles = glob.glob(datapath+'/*.dat')
    fringCorrFiles = glob.glob(datapath + '/LMC-058*_rfc1d.txt')

    # Creating an instance of the MRSstitch class
    stitcher = MRSstitch(fringCorrFiles)

    stitched_mrs_segments = stitcher.mrs_spectral_stitch()
    stitcher.plot_spectra(stitched_mrs_segments)

    # Trim the spectra to remove data after 28 microns and the save the final spectra
    final_spec_region = SpectralRegion(stitched_mrs_segments.spectral_axis[0], 28.03 * u.micron)
    finalspec = extract_region(stitched_mrs_segments, final_spec_region)

    outdatapath = "/Users/ojones/Science/SMPLMC58/"
    finalspec.write(outdatapath + "SMPLMC58_full.fits", overwrite=True)
