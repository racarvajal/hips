# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""HiPS tile drawing -- simple method."""
from typing import Generator, Any
import numpy as np
from astropy.wcs.utils import proj_plane_pixel_scales
from skimage.transform import ProjectiveTransform, warp
from ..tiles import HipsSurveyProperties, HipsTile, HipsTileMeta
from ..utils import WCSGeometry, compute_healpix_pixel_indices

__all__ = [
    'draw_sky_image',
    'make_sky_image',
    'SimpleTilePainter'
]


# TODO: Fix type annotation issue
def draw_sky_image(geometry: WCSGeometry, tiles: Generator[HipsTile, Any, Any], hips_survey: HipsSurveyProperties) -> np.ndarray:
    """Draw sky image using the simple and quick method.

    Parameters
    ----------
    geometry : `~hips.utils.WCSGeometry`
        An object of WCSGeometry
    tiles : List[HipsTile]
        A list of HipsTile
    hips_survey : `~hips.HipsSurveyProperties`
        HiPS survey properties

    Returns
    -------
    np.ndarray
        Returns a numpy array containing all HiPS tiles projected onto it
    """
    image = np.zeros(geometry.shape)
    for tile in tiles:
        painter = SimpleTilePainter(geometry, hips_survey, tile)
        image += painter.warp_image()
    return image


class SimpleTilePainter:
    """Paint a single tile using a simple projective transformation method.

    The algorithm implemented is described here: :ref:`drawing_algo`.

    Parameters
    ----------
    geometry : `~hips.utils.WCSGeometry`
        An object of WCSGeometry
    hips_survey : `~hips.HipsSurveyProperties`
        HiPS survey properties
    tile : `HipsTile`
       An object of HipsTile
    """

    def __init__(self, geometry: WCSGeometry, hips_survey: HipsSurveyProperties, tile: HipsTile) -> None:
        self.geometry = geometry
        self.hips_survey = hips_survey
        self.tile = tile

    @property
    def dst(self) -> np.ndarray:
        """Destination array for projective transform"""
        width = self.hips_survey.tile_width
        return np.array(
            [[width - 1, 0],
             [width - 1, width - 1],
             [0, width - 1],
             [0, 0]],
        )
    @property
    def projection(self) -> ProjectiveTransform:
        """Estimate projective transformation on a HiPS tile"""
        corners = self.tile.meta.skycoord_corners.to_pixel(self.geometry.wcs)
        src = np.array(corners).T.reshape((4, 2))
        dst = self.dst
        pt = ProjectiveTransform()
        pt.estimate(src, dst)
        return pt

    def warp_image(self) -> np.ndarray:
        """Warp a HiPS tile and a sky image"""
        return warp(
            self.tile.data,
            self.projection,
            output_shape=self.geometry.shape,
            preserve_range=True,
        )


def fetch_tiles(healpix_pixel_indices: np.ndarray, order: int, hips_survey: HipsSurveyProperties) -> 'HipsTile':
    """Fetch HiPS tiles from a remote URL.

    Parameters
    ----------
    healpix_pixel_indices : np.ndarray
        A list of HEALPix pixel indices
    order : int
        Order of the HEALPix map
    hips_survey : HipsSurveyProperties
        An object of HipsSurveyProperties

    Returns
    -------
    'HipsTile'
        Returns an object of  HipsTile
    """
    for healpix_pixel_index in healpix_pixel_indices:
        tile_meta = HipsTileMeta(
            order=order,
            ipix=healpix_pixel_index,
            frame=hips_survey.astropy_frame,
            file_format='fits',
        )
        tile = HipsTile.fetch(tile_meta, hips_survey.tile_access_url(order=order, ipix=healpix_pixel_index) + tile_meta.filename)
        yield tile


def get_order(geometry: WCSGeometry, hips_survey: HipsSurveyProperties) -> int:
    """Compute the tile order suited for the given geometry and hips_survey"""

    resolution = np.min(proj_plane_pixel_scales(geometry.wcs))

    tile_order = np.log2(hips_survey.tile_width)
    full_sphere_area = 4 * np.pi * np.square(180 / np.pi)
    for candidate_tile_order in range(3, 29 + 1):
        tile_resolution = np.sqrt(full_sphere_area / 12 / 4**(candidate_tile_order + tile_order))

        if tile_resolution <= resolution:
            break

    return np.min([candidate_tile_order, hips_survey.hips_order])


def make_sky_image(geometry: WCSGeometry, hips_survey: HipsSurveyProperties) -> np.ndarray:
    """Make sky image: fetch tiles and draw.

    The example for this can be found on the :ref:`gs` page.

    Parameters
    ----------
    geometry : `~hips.utils.WCSGeometry`
        Geometry of the output image
    hips_survey : `~hips.HipsSurveyProperties`
        HiPS survey properties

    Returns
    -------
    data : `~numpy.ndarray`
        Output image pixels
    """
    order = get_order(geometry, hips_survey)
    healpix_pixel_indices = compute_healpix_pixel_indices(
        wcs_geometry=geometry,
        order=order,
        healpix_frame=hips_survey.astropy_frame,
    )
    # TODO: this isn't a good API. Will become better when we have a cache.
    tiles = fetch_tiles(healpix_pixel_indices, order, hips_survey)

    image_data = draw_sky_image(geometry, tiles, hips_survey)

    return image_data
