# Electron Missions

This is used to Emulate Admin Gui rules via Kogs. Should only be used for missions that can not be 100 % migrated to Kogs. Can be thought of as a 4th DAS system. 


## Setup needed in Kogs

For this to work there must be matching rule in Kogni, e.g. there must be a corresponding mission profile rule for the customer / antenna / spacecraft combo.



## Matching rules


%customer -> %station -> %system -> %spacecraft

%customer -> %station -> %spacecraft

%customer -> %spacecraft

%spacecraft


### Example


KSAT TEST - Aqua @ MP1


## How the contact config is built

When the corresponding mission config is found, this is merged with the parameters found in the system config file. All parameters can be overwritten in the mission config file, so complete controll can be achived by creating a system specific file for the particular system / customer / spacecraft combo.


## Guis

(Hosted at the controller vm)

There are 3 GUIs that you can use: Grafana(port 3000), Influx(port 4000) and a simple GUI to send a few commands to the ACU(port 80) on supported systems. The Grafana dashboards can be created by the users. Changes and new dashboards should be checked into git, and they will be automatically synced with all antennas. Per now these guis are reachable over http but this will change to https in the future.

E.G. HA1: http://10.131.127.213:3000, http://10.131.127.213:4000, http://10.131.127.213