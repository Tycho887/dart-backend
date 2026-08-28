# Spacecrafts

Waiting for Spacecraft data from Kogs, this is a local database with spacecraft info(e.g. frequncies)

## Explanation of naming convention:

Each link gets a unique name that can be used as a reference for applications. The convention follows this schema: {$BAND}_band_{$DIRECTION}_p{$LINK_COUNTER}_{POLARIZATION_COUNTER}

The polarization counter is a bit strange, but is done to allow to reference a particular polarization when running a dual pol link on a single pol antenna. This means that if we have a different frequency then the physical link counter should be incremented, but if its the same frequency but dual polarization the polarization counter should be incremented.

