# Licensed under a 3-clause BSD style license - see LICENSE.rst
import asyncio
import requests
import concurrent.futures
from requests.exceptions import HTTPError, Timeout, ConnectionError
from typing import List
from ..tiles import HipsSurveyProperties, HipsTile, HipsTileMeta

__all__ = [
    'fetch_tiles',
]

__doctest_skip__ = [
    'fetch_tiles',
]


def fetch_tiles(tile_metas: List[HipsTileMeta], hips_survey: HipsSurveyProperties,
                progress_bar: bool = True, n_parallel: int = 5,
                timeout: float = 10, fetch_package: str = 'requests') -> List[HipsTile]:
    """Fetch a list of HiPS tiles.

    This function fetches a list of HiPS tiles based
    on their URLs, which are generated using ``hips_survey``
    and ``tile_metas``.

    The tiles are then fetched asynchronously using ``urllib`` or ``aiohttp``.

    Parameters
    ----------
    tile_metas : list
        Python list of `~hips.HipsTileMeta`
    hips_survey : `~hips.HipsSurveyProperties`
        HiPS survey properties
    progress_bar : bool
        Show a progress bar for tile fetching and drawing
    n_parallel : int
        Number of tile fetch web requests to make in parallel
    timeout : float
        Seconds to timeout for fetching a HiPS tile
    fetch_package : {'urllib', 'aiohttp'}
        Package to use for fetching HiPS tiles

    Examples
    --------
    Define a list of tiles we want::

        from hips import HipsSurveyProperties, HipsTileMeta
        from hips import fetch_tiles
        url = 'http://alasky.unistra.fr/DSS/DSS2Merged/properties'
        hips_survey = HipsSurveyProperties.fetch(url)
        tile_indices = [69623, 69627, 69628, 69629, 69630, 69631]
        tile_metas = []
        for healpix_pixel_index in tile_indices:
            tile_meta = HipsTileMeta(
               order=7,
               ipix=healpix_pixel_index,
               frame=hips_survey.astropy_frame,
               file_format='fits',
            )
            tile_metas.append(tile_meta)

    Fetch all tiles (in parallel)::

        tiles = fetch_tiles(tile_metas, hips_survey)

    Returns
    -------
    tiles : list
        A Python list of `~hips.HipsTile`
    """
    if fetch_package == 'aiohttp':
        fetch_fct = tiles_aiohttp
    elif fetch_package in ['urllib', 'requests']:
        fetch_fct = tiles_requests
    else:
        raise ValueError(f'Invalid package name: {fetch_package}')

    tiles = fetch_fct(tile_metas, hips_survey, progress_bar, n_parallel, timeout)

    out = []
    for tile_meta in tile_metas:
        for tile in tiles:
            if tile.meta == tile_meta:
                out.append(tile)
                continue
    return out


def fetch_tile_requests(url: str, meta: HipsTileMeta, timeout: float) -> HipsTile:
    """Fetch a HiPS tile using requests with proper headers and error handling."""
    headers = {
        "User-Agent": "hips-fetcher/1.0",
        "Accept": "image/fits;q=1.0, */*;q=0.1"
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return HipsTile(meta, response.content)
    except (HTTPError, Timeout, ConnectionError) as e:
        print(f"Error fetching {url}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching {url}: {e}")
        return None

def tiles_requests(tile_metas: List[HipsTileMeta], hips_survey: HipsSurveyProperties,
                   progress_bar: bool, n_parallel: int, timeout: float) -> List[HipsTile]:
    """Fetch HiPS tiles in parallel using requests."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = [executor.submit(fetch_tile_requests, hips_survey.tile_url(meta), meta, timeout) for meta in tile_metas]
        if progress_bar:
            from tqdm import tqdm
            futures = tqdm(concurrent.futures.as_completed(futures), total=len(tile_metas), desc='Fetching tiles')
        tiles = [future.result() for future in futures if future.result() is not None]
    if not tiles:
        raise ValueError("No HiPS tiles could be fetched. Check the HiPS server URL and availability.")
    return tiles

async def fetch_tile_aiohttp(url: str, meta: HipsTileMeta, session, timeout: float) -> HipsTile:
    """Fetch a HiPS tile asynchronously using aiohttp."""
    headers = {
        "User-Agent": "hips-fetcher/1.0",
        "Accept": "image/fits;q=1.0, */*;q=0.1"
    }
    try:
        async with session.get(url, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            raw_data = await response.read()
            return HipsTile(meta, raw_data)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

async def fetch_all_tiles_aiohttp(tile_metas: List[HipsTileMeta], hips_survey: HipsSurveyProperties,
                                  progress_bar: bool, n_parallel: int, timeout: float) -> List[HipsTile]:
    """Fetch HiPS tiles using aiohttp."""
    import aiohttp
    connector = aiohttp.TCPConnector(limit=n_parallel)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_tile_aiohttp(hips_survey.tile_url(meta), meta, session, timeout) for meta in tile_metas]
        if progress_bar:
            from tqdm.asyncio import tqdm
            tasks = tqdm(asyncio.as_completed(tasks), total=len(tile_metas), desc='Fetching tiles')
        tiles = [await task for task in tasks if task is not None]
    return tiles

def tiles_aiohttp(tile_metas: List[HipsTileMeta], hips_survey: HipsSurveyProperties,
                  progress_bar: bool, n_parallel: int, timeout: float) -> List[HipsTile]:
    return asyncio.run(fetch_all_tiles_aiohttp(tile_metas, hips_survey, progress_bar, n_parallel, timeout))
