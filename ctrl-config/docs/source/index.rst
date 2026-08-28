.. pylite documentation master file, created by
   sphinx-quickstart on Wed Mar  4 12:24:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mission configs's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   proxy

Source code is `here <https://gitlab.ksat.no/lite/configs/ctrl-config>`_

UPDATE
######
UPDATE 26-06-2020: New mission files should use v2. Old missions will be moved to this as far as possible. Check Admin GUI under MEOS config which is in use. 

General Notes
#############

These files are used when using ctrl_vX_* configuation files in AdminGui. It will look for a mission file that is named the same as the satellite name(except that - -> _, e.g ICEYE-X2 -> ICEYE_X2) Behave as playbooks that attempts to set backends/antennas in a certain state.

To create a mission file we need to know what equipment that are being used:

* proxy
* ctx
* specnet
* qradio
* qfep
* qmr
* qdra
* datadefender

Next how data are to be transfered for ctx, qfep, qdra(ftp, streaming etc). For customers connecting to backend we need to get the customer subnet from IT when they set up VPN, so we can whitelist this in the proxy.

Mission files, v2
#################

These files should replace v1, and hopefully be simpler and with more functionality. The main changes are simplified files, and how they work. A mission is setup by combing the v2/system/ANTENNA.yml and v2/mission/MISSION.yml(or v2/mission/SITE/ANTENNA/MISSION.yml if exists) into one template for each pass. The defaults section in v2/system/ANTENNA.yml can be overridden in the mission service blocks.

.. code-block:: yaml
		
		key: option1 | option2 .. | optionN [defaultValue] # Examples below have following syntax
				
If you want to use default value you do not need to add this to the file	
		
.. code-block:: yaml

		kind: mission | minimal [mission]
	
This use mission for standard mission. Minimal does not calculate predictions, check antenna tracking etc

.. code-block:: yaml

		mode: production | test [production]

.. code-block:: yaml

		config: string

Name of global config to load on backends

Services, v2
############

General notes:
Tags are used to decide which blocks are executed during run-time. Input tags are decided by choosing a config in Admin Gui, examples: 

ctrl_v2_usx -> tags: SU SD XD

ctrl_v2_us -> tags: SU SD

ctrl_v2_x -> tags: XD

.. code-block:: yaml
		
		tags:
		  - XD # Will be run by ctrl_v2_usx, ctrl_v2_x
		  - SD # Will be run by ctrl_v2_usx, ctrl_v2_us, ctrl_v2_s
		  - SU # Will be run by ctrl_v2_usx, ctrl_v2_us

proxy
*****

.. code-block:: yaml

		tags: tags as defined above, as list
		whitelist: list of ip addresses and subnets that are allowed to connect through the proxy during contact
		services: list of frontends that are opened, see a haproxy.cfg as sample

Example, this will open the proxy between AOS - LOS for 192.146.238.172 for the qRadio ports

.. code-block:: yaml

		tags:
		  - SD

		whitelist:
		  - 192.146.238.172

		services:
		  - qradio_tcp
		    		    
specnet
*******

.. code-block:: yaml

		tags: tags as defined above, as list
		config: string # Use this if you want to load a different config on specnet, than is defined globally
		rest: str on http://host:port format # Use this if you want to override default hosts. Use this only if needed and you know what you are doing, and never in a global config.

Example, this will load a global config on specnet

.. code-block:: yaml

		tags:
		  - SD
		  - SU

Example, this will load a specific config on specnet

.. code-block:: yaml

		tags:
		  - SD
		  - SU

		config: myconfig

		
				    
Mission files, v1
#################

UPDATE 26-06-2020: New mission files should use v2. Old missions will be moved to this as far as possible. Check Admin GUI under MEOS config which is in use. 

How mission files work. Start with high levels. Compare with actual mission files when reading the examples below.

.. code-block:: yaml
		
		key: option1 | option2 .. | optionN [defaultValue] # Examples below have following syntax

Above is template for examples below.
		
.. code-block:: yaml
		
		type: mission | minimal [mission]

This use mission for standard mission. Minimal does not calculate predictions, check antenna tracking etc

.. code-block:: yaml
		
		no_antenna: True | False [False]

Disables antenna features, should not be set to True unless you have a good reason.

.. code-block:: yaml
		
		config: missionConfig [missionFileName]

This is the default name of config that is loaded on equipment unless overridden in services blocks. If not present it would use the name of the mission file.

.. code-block:: yaml
		
		load_config: True|False [True]

If False config is not loaded on any of the blocks defined in services
		
.. code-block:: yaml
		
		hpa_watchdog: True|False [True]


If False the HPA watchdog is disabled. Might be relavant in some leop situations, but should by default be set to True


.. code-block:: yaml
		
		customer: customerName [None]
		constellation: constellationName [None]

Mostly metadata used for grouping. Important to include this, use KNOS names


Services, v1
############

General notes:
Tags are used to decide which blocks are executed during run-time. Input tags are decided by choosing a config in Admin Gui, examples:

ctrl_v1_usx -> tags: U S X

ctrl_v1_us -> tags: U S

ctrl_v1_x -> tags: X

.. code-block:: yaml
		
		tags:
		  - X # Will be run by ctrl_v1_usx, ctrl_v1_x
		  - S # Will be run by ctrl_v1_usx, ctrl_v1_us
		  - U # Will be run by ctrl_v1_usx, ctrl_v1_us

Tags need only 1 match		    
		    

Below the most common services are described

Types:
  configName: string
  
  bool: True | False
  
  restEndpoint: protocol://host:port | protocol://user:pw@host:port

proxy
*****

.. code-block:: yaml
		
		whitelist: list of ip addresses and subnets that are allowed to connect through the proxy during contact
		services: list of frontends that are opened, see a haproxy.cfg as sample


ctx
***

.. code-block:: yaml

		storage:
		  record: True | False [True] # Enable TM recording using ctx_storage. Data recorded to /data/ctx_storage @ scheduler
		  filename_flavour: legacy | std [std] # legacy filenames, should not be used in new missions

		transfer:
		  ftp:
		    host: 192.146.239.4 | 172.23.99.11 # ftp Tromsø or ftp Svalbard
		    port: 22 | 2222 # ftp Tromsø or ftp Svalbard
		    user: lite 
		    home: customerFtpHomeFolder ## Get this from PM, see other mission files


		 reset: True | False [True] # Reset CTX ports after LOS, should be set to True unless you have a reason not to

specnet
*******

.. code-block:: yaml

		config: configName [None] # Use this if you want to load a different config on specnet, than is defined globally
		rest: restEndpoint [None] # Use this if you want to override default hosts. Use this only if needed and you know what you are doing, and never in a global config.
		ctrl-agent: restEndpoint [None] # Use this if you use the above setting. 
		
		

qradio
******

Doppler files (in m/s, not hz) are generated automatically for all missions that uses this service. To enable the qradio config must be set up to support this. 

.. code-block:: yaml

		config: configName [None] # Use this if you want to load a different config on specnet, than is defined globally		


qfep
****

.. code-block:: yaml

		config: configName [None] # Use this if you want to load a different config on specnet, than is defined globally		
		

qmr
***

.. code-block:: yaml

		config: configName [None] # Use this if you want to load a different config on specnet, than is defined globally		
		

qdra-inst1
**********

.. code-block:: yaml

   		config: configName [None] # Use this if you want to load a different config on specnet, than is defined globally

		record:
		  streams: # List of channels on qdra
		    - raw

		  transfer:
		    ftp:
		      host: 192.146.239.4 | 172.23.99.11 # ftp Tromsø or ftp Svalbard
		      port: 22 | 2222 # ftp Tromsø or ftp Svalbard
		      user: lite 
		      home: customerFtpHomeFolder ## Get this from PM, see other mission files

		stream: True | False [False] ## Try avoid using this at the moment, unstable implemented from RTL. Also implications if file is not finished streaming at LOS. Streams data using playback

		  		
qdra-inst2
**********

Takes in same args as qdra-inst1, but talks to qdra-inst2, port 8081 instead of inst1. Using inst2 is not tested as much as inst1 as off 10 march 2020, so using this should involve testing before it is set operational.

Example - Full example
###############################

.. literalinclude:: examples/full_example.yml
   :language: yaml
   :linenos:
	      

	  
	   	
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
