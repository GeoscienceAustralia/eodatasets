import click

from ..prepare.s2_prepare_cophub_zip import main as s2_scihub
from ..prepare.ls_usgs_l1_prepare import main as ls_usgs
from ..prepare.s2_l1c_aws_pds_prepare import main as s2_awspds
from ..prepare.nasa_c_m_mcd43a1_6_prepare import main as mcd43a1
from ..prepare.noaa_c_c_prwtreatm_1_prepare import main as prwtr


@click.group()
@click.version_option()
def run():
    pass


run.add_command(s2_scihub, name='s2-cophub')
run.add_command(ls_usgs, name='ls-usgs')
run.add_command(s2_awspds, name='s2-awspds')
run.add_command(mcd43a1, name='modis-mcd43a1')
run.add_command(prwtr, name='noaa-prwtr')


if __name__ == '__main__':
    run()
