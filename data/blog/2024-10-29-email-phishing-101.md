---
layout: post
title: 'How to Launch Your First Email Phishing Drill for Effective Cybersecurity Training'
subtitle: 'Comprehensive Guide to Implementing Gophish and SMTP Email Services for Phishing Simulations'
tags:
  - Pentest
  - Social-Engineering
  - Email-Phishing
  - Cybersecurity-Training
  - Gophish
  - SMTP-Services
---

## What is Email Phishing Drill

An email phishing drill is a simulated cyber attack training method designed to enhance employees' ability to recognize and respond to phishing attacks. In such drills, organizations use tools like Gophish and SMTP services to send mock phishing emails, testing employees' reactions to potential threats. This not only helps employees identify common characteristics of phishing emails but also raises overall cybersecurity awareness. By conducting regular phishing drills, companies can effectively reduce the risk of data breaches and security incidents caused by phishing attacks.

## Prerequisite

- A VPS from CSP (i.e. cloud service provider, it could be AWS/Azure/GCP) or any Ubuntu
  machine that can be accessed from internet.
- A domain. (you can select one in websites like godaddy or purchase it from CSP)
- [Optional] AWS SES (simple email services)

## Step 1 - Install Gophish

I personally recommand to install gophish directly in your machine's path, rather than to install it in docker, since you have to modify your config file often.

The very first step is to clone the `gophish` project.

**Notes: check your version!**

```
wget https://github.com/gophish/gophish/releases/download/v0.12.1/gophish-v0.12.1-linux-64bit.zip
```

Then unzip it and you can now enter the gophish folder:

```shell
unzip gophish-v0.12.1.linux-64bit.zip
cd gophish-v0.12.1
```

Now you can run `./gophish` to start gophish, but we have a better way where we can use `systemctl` to manage the start and stop of gophish service. What you have to do is to create a file `/etc/systemd/system/gophish.service` using:

```shell
vim /etc/systemd/system/gophish.service
```

And enter the content:

```
[Unit]
Description=Gophish Service
After=network.target

[Service]
Type=simple
ExecStart=/home/ubuntu/gophish-v0.12.1/gophish
WorkingDirectory=/home/ubuntu/gophish-v0.12.1
User=root
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Now you can use following command:

```shell
sudo systemctl enable gophish
sudo systemctl start gophish
sudo systemctl stop gophish
sudo systemctl status gophish
```

## Step 2 - Config Gophish

Now go back to the root path of your gophish, we will create licenses for our domain:

```shell
sudo apt update
sudo apt install certbot
sudo certbot certonly --manual --preferred-challenges dns -d example.com
```

This will give you a DNS config requirements, and you need to set following DNS record like this.

![image-20241029154448353](../img/in-post/2024-10-29-email-phishing-101/image-20241029154448353.png)

**More over, you have to set an A records to let your domain point to your server's IPv4.**

After the certification of your DNS record, you will be given two files in `/etc/letsencrypt/live/example.com/`

Now, edit your `config.js` in gophish root path with following content

```
{
    "admin_server": {
        "listen_url": "0.0.0.0:3333",
        "use_tls": true,
        "cert_path": "/etc/letsencrypt/live/example.com/fullchain.pem",
        "key_path": "/etc/letsencrypt/live/example.com/privkey.pem",
        "trusted_origins": []
    },
    "phish_server": {
        "listen_url": "0.0.0.0:443",
        "use_tls": true,
        "cert_path": "/etc/letsencrypt/live/example.com/fullchain.pem",
        "key_path": "/etc/letsencrypt/live/example.com/privkey.pem"
    },
    "db_name": "sqlite3",
    "db_path": "gophish.db",
    "migrations_prefix": "db/db_",
    "contact_address": "support@example.com",
    "logging": {
        "filename": "/var/log/gophish/gophish.log",
        "level": ""
    }
}

```

## Step 3 - Start Gophish

After running `sudo systemctl start gophish` you now be able to visit `https://example.com:3333`. Your password is in your console output, you can visit logs:

```
cat /var/log/gophish/gophish.log
```

like this:

```
gophish@gophish.dev:~/src/github.com/gophish/gophish$ ./gophish
 time="2020-06-30T08:04:33-05:00" level=warning msg="No contact address has been configured."
 time="2020-06-30T08:04:33-05:00" level=warning msg="Please consider adding a contact_address entry in your config.json"
 time="2020-06-30T08:04:33-05:00" level=info msg="Please login with the username admin and the password 1178f855283d03d3"
 time="2020-06-30T08:04:33-05:00" level=info msg="Starting phishing server at http://0.0.0.0:80"
 time="2020-06-30T08:04:33-05:00" level=info msg="Starting IMAP monitor manager"
 time="2020-06-30T08:04:33-05:00" level=info msg="Starting admin server at https://127.0.0.1:3333"
 time="2020-06-30T08:04:33-05:00" level=info msg="Background Worker Started Successfully - Waiting for Campaigns"
 time="2020-06-30T08:04:33-05:00" level=info msg="Starting new IMAP monitor for user admin"
```

## Step 5- Get Fake Email and Fake Landing Page.

You can use 'import email' to load raw email data, and any links in the email will be replaced. Or if you write emails yourself, use `.URL`:

```html
<a href=".URL">somenormallinks.com</a>
```

As for landing page, your landing page must satisfy following html structure, you may have to disable origin submit button.

```html
<form>
  <input />
  <input />
  <input type="submit" />
</form>
```

The landing page will be served once your start a campaign, in `https://example.com/?rid={randomkey}`. You can put static recourses file in:

```
~/gophish-v0.12.1/static
```

For exmaple:

```
~/gophish-v0.12.1/static/images/logo.png
```

To access it , use `https://example.com/static/logo.png`

## Step 6 Get SMTP services

### AWS SES

The easiest way to get an SMTP services is to use AWS SES, in free-tier, you can send less than 200 emails in a day in sandbox mode, which means only certified domains/email addresses can be used to send/receive emails. You can select `Create SMTP credentials` and follow the instructure to get your connection credentials. Notes: Do not use 25 since most CSP forbidden 25 connection and 25 is SMTP without encryption. Use other ports when using SES.

> We will be using 25 soon, in our own SMTP.

![image-20241029162955857](../img/in-post/2024-10-29-email-phishing-101/image-20241029162955857.png)

You might create a sending profile in gophish like this:

```
SMTP from : someone@example.com
Host: email-smtp.us-east-1.amazonaws.com:587
Username: AKIAZJGUWXXXXXXXXXXX
Password: xxxxxxxxxxxxxxxxxxxxxxxxx
```

### Custom SMTP

You can deploy SMTP in same machine or another one. I recommend to deploy it in another machine and use balance load or other product to redirect port 25. You can use following design:

```mermaid
graph LR
    A[Ubuntu with Gophish] -->|Request to 257| B[Load Balancer]
    B -->|Request to 25| C[Internal SMTP Server]
    C -->|Send Email| D[Other Internal Servers]

```

Now in your SMTP server, install `docker-mailserver`.

First, create a `mailserver.env`, the only diff in this `env` file is that we enable SMTP ONLY.

```
# -----------------------------------------------
# --- Mailserver Environment Variables ----------
# -----------------------------------------------

# DOCUMENTATION FOR THESE VARIABLES IS FOUND UNDER
# https://docker-mailserver.github.io/docker-mailserver/latest/config/environment/

# -----------------------------------------------
# --- General Section ---------------------------
# -----------------------------------------------

# empty => uses the `hostname` command to get the mail server's canonical hostname
# => Specify a fully-qualified domainname to serve mail for.  This is used for many of the config features so if you can't set your hostname (e.g. you're in a container platform that doesn't let you) specify it in this environment variable.
OVERRIDE_HOSTNAME=

# REMOVED in version v11.0.0! Use LOG_LEVEL instead.
DMS_DEBUG=0

# Set the log level for DMS.
# This is mostly relevant for container startup scripts and change detection event feedback.
#
# Valid values (in order of increasing verbosity) are: `error`, `warn`, `info`, `debug` and `trace`.
# The default log level is `info`.
LOG_LEVEL=info

# critical => Only show critical messages
# error => Only show erroneous output
# **warn** => Show warnings
# info => Normal informational output
# debug => Also show debug messages
SUPERVISOR_LOGLEVEL=

# Support for deployment where these defaults are not compatible (eg: some NAS appliances):
# /var/mail vmail User ID (default: 5000)
DMS_VMAIL_UID=
# /var/mail vmail Group ID (default: 5000)
DMS_VMAIL_GID=

# **empty** => use FILE
# LDAP => use LDAP authentication
# OIDC => use OIDC authentication (not yet implemented)
# FILE => use local files (this is used as the default)
ACCOUNT_PROVISIONER=

# empty => postmaster@domain.com
# => Specify the postmaster address
POSTMASTER_ADDRESS=

# Check for updates on container start and then once a day
# If an update is available, a mail is sent to POSTMASTER_ADDRESS
# 0 => Update check disabled
# 1 => Update check enabled
ENABLE_UPDATE_CHECK=1

# Customize the update check interval.
# Number + Suffix. Suffix must be 's' for seconds, 'm' for minutes, 'h' for hours or 'd' for days.
UPDATE_CHECK_INTERVAL=1d

# Set different options for mynetworks option (can be overwrite in postfix-main.cf)
# **WARNING**: Adding the docker network's gateway to the list of trusted hosts, e.g. using the `network` or
# `connected-networks` option, can create an open relay
# https://github.com/docker-mailserver/docker-mailserver/issues/1405#issuecomment-590106498
# The same can happen for rootless podman. To prevent this, set the value to "none" or configure slirp4netns
# https://github.com/docker-mailserver/docker-mailserver/issues/2377
#
# none => Explicitly force authentication
# container => Container IP address only
# host => Add docker container network (ipv4 only)
# network => Add all docker container networks (ipv4 only)
# connected-networks => Add all connected docker networks (ipv4 only)
PERMIT_DOCKER=none

# Set the timezone. If this variable is unset, the container runtime will try to detect the time using
# `/etc/localtime`, which you can alternatively mount into the container. The value of this variable
# must follow the pattern `AREA/ZONE`, i.e. of you want to use Germany's time zone, use `Europe/Berlin`.
# You can lookup all available timezones here: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones#List
TZ=

# In case you network interface differs from 'eth0', e.g. when you are using HostNetworking in Kubernetes,
# you can set NETWORK_INTERFACE to whatever interface you want. This interface will then be used.
#  - **empty** => eth0
NETWORK_INTERFACE=

# empty => modern
# modern => Enables TLSv1.2 and modern ciphers only. (default)
# intermediate => Enables TLSv1, TLSv1.1 and TLSv1.2 and broad compatibility ciphers.
TLS_LEVEL=

# Configures the handling of creating mails with forged sender addresses.
#
# **0** => (not recommended) Mail address spoofing allowed. Any logged in user may create email messages with a forged sender address (see also https://en.wikipedia.org/wiki/Email_spoofing).
# 1 => Mail spoofing denied. Each user may only send with his own or his alias addresses. Addresses with extension delimiters(http://www.postfix.org/postconf.5.html#recipient_delimiter) are not able to send messages.
SPOOF_PROTECTION=

# Enables the Sender Rewriting Scheme. SRS is needed if your mail server acts as forwarder. See [postsrsd](https://github.com/roehling/postsrsd/blob/master/README.md#sender-rewriting-scheme-crash-course) for further explanation.
#  - **0** => Disabled
#  - 1 => Enabled
ENABLE_SRS=0

# Enables the OpenDKIM service.
# **1** => Enabled
#   0   => Disabled
ENABLE_OPENDKIM=1

# Enables the OpenDMARC service.
# **1** => Enabled
#   0   => Disabled
ENABLE_OPENDMARC=1


# Enabled `policyd-spf` in Postfix's configuration. You will likely want to set this
# to `0` in case you're using Rspamd (`ENABLE_RSPAMD=1`).
#
# - 0     => Disabled
# - **1** => Enabled
ENABLE_POLICYD_SPF=1

# Enables POP3 service
# - **0** => Disabled
# - 1     => Enabled
ENABLE_POP3=

# Enables IMAP service
# - 0     => Disabled
# - **1** => Enabled
ENABLE_IMAP=1

# Enables ClamAV, and anti-virus scanner.
#   1   => Enabled
# **0** => Disabled
ENABLE_CLAMAV=0

# Add the value of this ENV as a prefix to the mail subject when spam is detected.
# NOTE: This subject prefix may be redundant (by default spam is delivered to a junk folder).
#       It provides value when your junk mail is stored alongside legitimate mail instead of a separate location (like with `SPAMASSASSIN_SPAM_TO_INBOX=1` or `MOVE_SPAM_TO_JUNK=0` or a POP3 only setup, without IMAP).
# NOTE: When not using Docker Compose, other CRI may not support quote-wrapping the value here to preserve any trailing white-space.
SPAM_SUBJECT=

# Enables Rspamd
# **0** => Disabled
#   1   => Enabled
ENABLE_RSPAMD=0

# When `ENABLE_RSPAMD=1`, an internal Redis instance is enabled implicitly.
# This setting provides an opt-out to allow using an external instance instead.
# 0 => Disabled
# 1 => Enabled
ENABLE_RSPAMD_REDIS=

# When enabled,
#
# 1. the "[autolearning][rspamd-autolearn]" feature is turned on;
# 2. the Bayes classifier will be trained when moving mails from or to the Junk folder (with the help of Sieve scripts).
#
# **0** => disabled
# 1     => enabled
RSPAMD_LEARN=0

# This settings controls whether checks should be performed on emails coming
# from authenticated users (i.e. most likely outgoing emails). The default value
# is `0` in order to align better with SpamAssassin. We recommend reading
# through https://rspamd.com/doc/tutorials/scanning_outbound.html though to
# decide for yourself whether you need and want this feature.
#
# Note that DKIM signing of e-mails will still happen.
RSPAMD_CHECK_AUTHENTICATED=0

# Controls whether the Rspamd Greylisting module is enabled.
# This module can further assist in avoiding spam emails by greylisting
# e-mails with a certain spam score.
#
# **0** => disabled
# 1     => enabled
RSPAMD_GREYLISTING=0

# Can be used to enable or disable the Hfilter group module.
#
# - 0     => Disabled
# - **1** => Enabled
RSPAMD_HFILTER=1

# Can be used to control the score when the HFILTER_HOSTNAME_UNKNOWN symbol applies. A higher score is more punishing. Setting it to 15 is equivalent to rejecting the email when the check fails.
#
# Default: 6
RSPAMD_HFILTER_HOSTNAME_UNKNOWN_SCORE=6

# Can be used to enable or disable the (still experimental) neural module.
#
# - **0** => Disabled
# - 1     => Enabled
RSPAMD_NEURAL=0

# Amavis content filter (used for ClamAV & SpamAssassin)
# 0 => Disabled
# 1 => Enabled
ENABLE_AMAVIS=1

# -1/-2/-3 => Only show errors
# **0**    => Show warnings
# 1/2      => Show default informational output
# 3/4/5    => log debug information (very verbose)
AMAVIS_LOGLEVEL=0

# This enables DNS block lists in Postscreen.
# Note: Emails will be rejected, if they don't pass the block list checks!
# **0** => DNS block lists are disabled
# 1     => DNS block lists are enabled
ENABLE_DNSBL=0

# If you enable Fail2Ban, don't forget to add the following lines to your `compose.yaml`:
#    cap_add:
#      - NET_ADMIN
# Otherwise, `nftables` won't be able to ban IPs.
ENABLE_FAIL2BAN=0

# Fail2Ban blocktype
# drop   => drop packet (send NO reply)
# reject => reject packet (send ICMP unreachable)
FAIL2BAN_BLOCKTYPE=drop

# 1 => Enables Managesieve on port 4190
# empty => disables Managesieve
ENABLE_MANAGESIEVE=

# **enforce** => Allow other tests to complete. Reject attempts to deliver mail with a 550 SMTP reply, and log the helo/sender/recipient information. Repeat this test the next time the client connects.
# drop => Drop the connection immediately with a 521 SMTP reply. Repeat this test the next time the client connects.
# ignore => Ignore the failure of this test. Allow other tests to complete. Repeat this test the next time the client connects. This option is useful for testing and collecting statistics without blocking mail.
POSTSCREEN_ACTION=enforce

# empty => all daemons start
# 1 => only launch postfix smtp
SMTP_ONLY=1


# Please read [the SSL page in the documentation](https://docker-mailserver.github.io/docker-mailserver/latest/config/security/ssl) for more information.
#
# empty => SSL disabled
# letsencrypt => Enables Let's Encrypt certificates
# custom => Enables custom certificates
# manual => Let's you manually specify locations of your SSL certificates for non-standard cases
# self-signed => Enables self-signed certificates
SSL_TYPE=letsencrypt

# These are only supported with `SSL_TYPE=manual`.
# Provide the path to your cert and key files that you've mounted access to within the container.
SSL_CERT_PATH=
SSL_KEY_PATH=
# Optional: A 2nd certificate can be supported as fallback (dual cert support), eg ECDSA with an RSA fallback.
# Useful for additional compatibility with older MTA and MUA (eg pre-2015).
SSL_ALT_CERT_PATH=
SSL_ALT_KEY_PATH=

# Set how many days a virusmail will stay on the server before being deleted
# empty => 7 days
VIRUSMAILS_DELETE_DELAY=

# Configure Postfix `virtual_transport` to deliver mail to a different LMTP client (default is a dovecot socket).
# Provide any valid URI. Examples:
#
# empty => `lmtp:unix:/var/run/dovecot/lmtp` (default, configured in Postfix main.cf)
# `lmtp:unix:private/dovecot-lmtp` (use socket)
# `lmtps:inet:<host>:<port>` (secure lmtp with starttls)
# `lmtp:<kopano-host>:2003` (use kopano as mailstore)
POSTFIX_DAGENT=

# Set the mailbox size limit for all users. If set to zero, the size will be unlimited (default). Size is in bytes.
#
# empty => 0
POSTFIX_MAILBOX_SIZE_LIMIT=

# See https://docker-mailserver.github.io/docker-mailserver/latest/config/account-management/overview/#quotas
# 0 => Dovecot quota is disabled
# 1 => Dovecot quota is enabled
ENABLE_QUOTAS=1

# Set the message size limit for all users. If set to zero, the size will be unlimited (not recommended!). Size is in bytes.
#
# empty => 10240000 (~10 MB)
POSTFIX_MESSAGE_SIZE_LIMIT=

# Mails larger than this limit won't be scanned.
# ClamAV must be enabled (ENABLE_CLAMAV=1) for this.
#
# empty => 25M (25 MB)
CLAMAV_MESSAGE_SIZE_LIMIT=

# Enables regular pflogsumm mail reports.
# This is a new option. The old REPORT options are still supported for backwards compatibility. If this is not set and reports are enabled with the old options, logrotate will be used.
#
# not set => No report
# daily_cron => Daily report for the previous day
# logrotate => Full report based on the mail log when it is rotated
PFLOGSUMM_TRIGGER=

# Recipient address for pflogsumm reports.
#
# not set => Use REPORT_RECIPIENT or POSTMASTER_ADDRESS
# => Specify the recipient address(es)
PFLOGSUMM_RECIPIENT=

# Sender address (`FROM`) for pflogsumm reports if pflogsumm reports are enabled.
#
# not set => Use REPORT_SENDER
# => Specify the sender address
PFLOGSUMM_SENDER=

# Interval for logwatch report.
#
# none => No report is generated
# daily => Send a daily report
# weekly => Send a report every week
LOGWATCH_INTERVAL=

# Recipient address for logwatch reports if they are enabled.
#
# not set => Use REPORT_RECIPIENT or POSTMASTER_ADDRESS
# => Specify the recipient address(es)
LOGWATCH_RECIPIENT=

# Sender address (`FROM`) for logwatch reports if logwatch reports are enabled.
#
# not set => Use REPORT_SENDER
# => Specify the sender address
LOGWATCH_SENDER=

# Defines who receives reports if they are enabled.
# **empty** => ${POSTMASTER_ADDRESS}
# => Specify the recipient address
REPORT_RECIPIENT=

# Defines who sends reports if they are enabled.
# **empty** => mailserver-report@${DOMAINNAME}
# => Specify the sender address
REPORT_SENDER=

# Changes the interval in which log files are rotated
# **weekly** => Rotate log files weekly
# daily => Rotate log files daily
# monthly => Rotate log files monthly
#
# Note: This Variable actually controls logrotate inside the container
# and rotates the log files depending on this setting. The main log output is
# still available in its entirety via `docker logs mail` (Or your
# respective container name). If you want to control logrotation for
# the Docker-generated logfile see:
# https://docs.docker.com/config/containers/logging/configure/
#
# Note: This variable can also determine the interval for Postfix's log summary reports, see [`PFLOGSUMM_TRIGGER`](#pflogsumm_trigger).
LOGROTATE_INTERVAL=weekly

# Defines how many log files are kept by logrorate
LOGROTATE_COUNT=4


# If enabled, employs `reject_unknown_client_hostname` to sender restrictions in Postfix's configuration.
#
# - **0** => Disabled
# - 1 => Enabled
POSTFIX_REJECT_UNKNOWN_CLIENT_HOSTNAME=0

# Choose TCP/IP protocols for postfix to use
# **all** => All possible protocols.
# ipv4 => Use only IPv4 traffic. Most likely you want this behind Docker.
# ipv6 => Use only IPv6 traffic.
#
# Note: More details at http://www.postfix.org/postconf.5.html#inet_protocols
POSTFIX_INET_PROTOCOLS=all

# Enables MTA-STS support for outbound mail.
# More details: https://docker-mailserver.github.io/docker-mailserver/v13.3/config/best-practices/mta-sts/
# - **0** ==> MTA-STS disabled
# - 1 => MTA-STS enabled
ENABLE_MTA_STS=0

# Choose TCP/IP protocols for dovecot to use
# **all** => Listen on all interfaces
# ipv4 => Listen only on IPv4 interfaces. Most likely you want this behind Docker.
# ipv6 => Listen only on IPv6 interfaces.
#
# Note: More information at https://dovecot.org/doc/dovecot-example.conf
DOVECOT_INET_PROTOCOLS=all

# -----------------------------------------------
# --- SpamAssassin Section ----------------------
# -----------------------------------------------

ENABLE_SPAMASSASSIN=0

# KAM is a 3rd party SpamAssassin ruleset, provided by the McGrail Foundation.
# If SpamAssassin is enabled, KAM can be used in addition to the default ruleset.
# - **0** => KAM disabled
# - 1 => KAM enabled
#
# Note: only has an effect if `ENABLE_SPAMASSASSIN=1`
ENABLE_SPAMASSASSIN_KAM=0

# deliver spam messages to the inbox (tagged using SPAM_SUBJECT)
SPAMASSASSIN_SPAM_TO_INBOX=1

# spam messages will be moved in the Junk folder (SPAMASSASSIN_SPAM_TO_INBOX=1 required)
MOVE_SPAM_TO_JUNK=1

# spam messages will be marked as read
MARK_SPAM_AS_READ=0

# add 'spam info' headers at, or above this level
SA_TAG=2.0

# add 'spam detected' headers at, or above this level
SA_TAG2=6.31

# triggers spam evasive actions
SA_KILL=10.0

# -----------------------------------------------
# --- Fetchmail Section -------------------------
# -----------------------------------------------

ENABLE_FETCHMAIL=0

# The interval to fetch mail in seconds
FETCHMAIL_POLL=300
# Use multiple fetchmail instances (1 per poll entry in fetchmail.cf)
# Supports multiple IMAP IDLE connections when a server is used across multiple poll entries
# https://otremba.net/wiki/Fetchmail_(Debian)#Immediate_Download_via_IMAP_IDLE
FETCHMAIL_PARALLEL=0

# Enable or disable `getmail`.
#
# - **0** => Disabled
# - 1 => Enabled
ENABLE_GETMAIL=0

# The number of minutes for the interval. Min: 1; Default: 5.
GETMAIL_POLL=5

# -----------------------------------------------
# --- OAUTH2 Section ----------------------------
# -----------------------------------------------

# empty => OAUTH2 authentication is disabled
# 1 => OAUTH2 authentication is enabled
ENABLE_OAUTH2=

# Specify the user info endpoint URL of the oauth2 provider
# Example: https://oauth2.example.com/userinfo/
OAUTH2_INTROSPECTION_URL=

# -----------------------------------------------
# --- LDAP Section ------------------------------
# -----------------------------------------------

# A second container for the ldap service is necessary (i.e. https://hub.docker.com/r/bitnami/openldap/)

# empty => no
# yes => LDAP over TLS enabled for Postfix
LDAP_START_TLS=

# empty => mail.example.com
# Specify the `<dns-name>` / `<ip-address>` where the LDAP server is reachable via a URI like: `ldaps://mail.example.com`.
# Note: You must include the desired URI scheme (`ldap://`, `ldaps://`, `ldapi://`).
LDAP_SERVER_HOST=

# empty => ou=people,dc=domain,dc=com
# => e.g. LDAP_SEARCH_BASE=dc=mydomain,dc=local
LDAP_SEARCH_BASE=

# empty => cn=admin,dc=domain,dc=com
# => take a look at examples of SASL_LDAP_BIND_DN
LDAP_BIND_DN=

# empty** => admin
# => Specify the password to bind against ldap
LDAP_BIND_PW=

# e.g. `"(&(mail=%s)(mailEnabled=TRUE))"`
# => Specify how ldap should be asked for users
LDAP_QUERY_FILTER_USER=

# e.g. `"(&(mailGroupMember=%s)(mailEnabled=TRUE))"`
# => Specify how ldap should be asked for groups
LDAP_QUERY_FILTER_GROUP=

# e.g. `"(&(mailAlias=%s)(mailEnabled=TRUE))"`
# => Specify how ldap should be asked for aliases
LDAP_QUERY_FILTER_ALIAS=

# e.g. `"(&(|(mail=*@%s)(mailalias=*@%s)(mailGroupMember=*@%s))(mailEnabled=TRUE))"`
# => Specify how ldap should be asked for domains
LDAP_QUERY_FILTER_DOMAIN=

# -----------------------------------------------
# --- Dovecot Section ---------------------------
# -----------------------------------------------

# empty => no
# yes => LDAP over TLS enabled for Dovecot
DOVECOT_TLS=

# e.g. `"(&(objectClass=PostfixBookMailAccount)(uniqueIdentifier=%n))"`
DOVECOT_USER_FILTER=

# e.g. `"(&(objectClass=PostfixBookMailAccount)(uniqueIdentifier=%n))"`
DOVECOT_PASS_FILTER=

# Define the mailbox format to be used
# default is maildir, supported values are: sdbox, mdbox, maildir
DOVECOT_MAILBOX_FORMAT=maildir

# empty => no
# yes => Allow bind authentication for LDAP
# https://wiki.dovecot.org/AuthDatabase/LDAP/AuthBinds
DOVECOT_AUTH_BIND=

# -----------------------------------------------
# --- Postgrey Section --------------------------
# -----------------------------------------------

ENABLE_POSTGREY=0
# greylist for N seconds
POSTGREY_DELAY=300
# delete entries older than N days since the last time that they have been seen
POSTGREY_MAX_AGE=35
# response when a mail is greylisted
POSTGREY_TEXT="Delayed by Postgrey"
# whitelist host after N successful deliveries (N=0 to disable whitelisting)
POSTGREY_AUTO_WHITELIST_CLIENTS=5

# -----------------------------------------------
# --- SASL Section ------------------------------
# -----------------------------------------------

ENABLE_SASLAUTHD=0

# empty => pam
# `ldap` => authenticate against ldap server
# `shadow` => authenticate against local user db
# `mysql` => authenticate against mysql db
# `rimap` => authenticate against imap server
# Note: can be a list of mechanisms like pam ldap shadow
SASLAUTHD_MECHANISMS=

# empty => None
# e.g. with SASLAUTHD_MECHANISMS rimap you need to specify the ip-address/servername of the imap server  ==> xxx.xxx.xxx.xxx
SASLAUTHD_MECH_OPTIONS=

# empty => Use value of LDAP_SERVER_HOST
# Note: You must include the desired URI scheme (`ldap://`, `ldaps://`, `ldapi://`).
SASLAUTHD_LDAP_SERVER=

# empty => Use value of LDAP_BIND_DN
# specify an object with privileges to search the directory tree
# e.g. active directory: SASLAUTHD_LDAP_BIND_DN=cn=Administrator,cn=Users,dc=mydomain,dc=net
# e.g. openldap: SASLAUTHD_LDAP_BIND_DN=cn=admin,dc=mydomain,dc=net
SASLAUTHD_LDAP_BIND_DN=

# empty => Use value of LDAP_BIND_PW
SASLAUTHD_LDAP_PASSWORD=

# empty => Use value of LDAP_SEARCH_BASE
# specify the search base
SASLAUTHD_LDAP_SEARCH_BASE=

# empty => default filter `(&(uniqueIdentifier=%u)(mailEnabled=TRUE))`
# e.g. for active directory: `(&(sAMAccountName=%U)(objectClass=person))`
# e.g. for openldap: `(&(uid=%U)(objectClass=person))`
SASLAUTHD_LDAP_FILTER=

# empty => no
# yes => LDAP over TLS enabled for SASL
# If set to yes, the protocol in SASLAUTHD_LDAP_SERVER must be ldap:// or missing.
SASLAUTHD_LDAP_START_TLS=

# empty => no
# yes => Require and verify server certificate
# If yes you must/could specify SASLAUTHD_LDAP_TLS_CACERT_FILE or SASLAUTHD_LDAP_TLS_CACERT_DIR.
SASLAUTHD_LDAP_TLS_CHECK_PEER=

# File containing CA (Certificate Authority) certificate(s).
# empty => Nothing is added to the configuration
# Any value => Fills the `ldap_tls_cacert_file` option
SASLAUTHD_LDAP_TLS_CACERT_FILE=

# Path to directory with CA (Certificate Authority) certificates.
# empty => Nothing is added to the configuration
# Any value => Fills the `ldap_tls_cacert_dir` option
SASLAUTHD_LDAP_TLS_CACERT_DIR=

# Specify what password attribute to use for password verification.
# empty => Nothing is added to the configuration but the documentation says it is `userPassword` by default.
# Any value => Fills the `ldap_password_attr` option
SASLAUTHD_LDAP_PASSWORD_ATTR=

# empty => `bind` will be used as a default value
# `fastbind` => The fastbind method is used
# `custom` => The custom method uses userPassword attribute to verify the password
SASLAUTHD_LDAP_AUTH_METHOD=

# Specify the authentication mechanism for SASL bind
# empty => Nothing is added to the configuration
# Any value => Fills the `ldap_mech` option
SASLAUTHD_LDAP_MECH=

# -----------------------------------------------
# --- SRS Section -------------------------------
# -----------------------------------------------

# envelope_sender => Rewrite only envelope sender address (default)
# header_sender => Rewrite only header sender (not recommended)
# envelope_sender,header_sender => Rewrite both senders
# An email has an "envelope" sender (indicating the sending server) and a
# "header" sender (indicating who sent it). More strict SPF policies may require
# you to replace both instead of just the envelope sender.
SRS_SENDER_CLASSES=envelope_sender

# empty => Envelope sender will be rewritten for all domains
# provide comma separated list of domains to exclude from rewriting
SRS_EXCLUDE_DOMAINS=

# empty => generated when the image is built
# provide a secret to use in base64
# you may specify multiple keys, comma separated. the first one is used for
# signing and the remaining will be used for verification. this is how you
# rotate and expire keys
SRS_SECRET=

# -----------------------------------------------
# --- Default Relay Host Section ----------------
# -----------------------------------------------

# Setup relaying all mail through a default relay host
#
# Set a default host to relay all mail through (optionally include a port)
# Example: [mail.example.com]:587
DEFAULT_RELAY_HOST=

# -----------------------------------------------
# --- Multi-Domain Relay Section ----------------
# -----------------------------------------------

# Setup relaying for multiple domains based on the domain name of the sender
# optionally uses usernames and passwords in postfix-sasl-password.cf and relay host mappings in postfix-relaymap.cf
#
# Set a default host to relay mail through
# Example: mail.example.com
RELAY_HOST=

# empty => 25
# default port to relay mail
RELAY_PORT=25

# -----------------------------------------------
# --- Relay Host Credentials Section ------------
# -----------------------------------------------

# Configure a relay user and password to use with RELAY_HOST / DEFAULT_RELAY_HOST

# empty => no default
RELAY_USER=

# empty => no default
RELAY_PASSWORD=

```

Then create following `docker-compose.yaml` and conduct `docker compose up`, make sure to replace the domain. In this compose file, we mount Certbot path and smtp service postfix config path.

```
services:
  mailserver:
    image: ghcr.io/docker-mailserver/docker-mailserver:latest
    container_name: mailserver
    # Provide the FQDN of your mail server here (Your DNS MX record should point to this value)
    hostname: example.com
    env_file: mailserver.env
    # More information about the mail-server ports:
    # https://docker-mailserver.github.io/docker-mailserver/latest/config/security/understanding-the-ports/
    ports:
      - "25:25"    # SMTP  (explicit TLS => STARTTLS, Authentication is DISABLED => use port 465/587 instead)
      - "143:143"  # IMAP4 (explicit TLS => STARTTLS)
      - "465:465"  # ESMTP (implicit TLS)
      - "587:587"  # ESMTP (explicit TLS => STARTTLS)
      - "993:993"  # IMAP4 (implicit TLS)
    volumes:
      - ./docker-data/dms/mail-data/:/var/mail/
      - ./docker-data/dms/mail-state/:/var/mail-state/
      - ./docker-data/dms/mail-logs/:/var/log/mail/
      - ./docker-data/dms/config/:/tmp/docker-mailserver/
      - /etc/localtime:/etc/localtime:ro
      - /etc/letsencrypt:/etc/letsencrypt
      - /etc/postfix/main.cf:/etc/postfix/main.cf
    restart: always
    stop_grace_period: 1m
    # Uncomment if using `ENABLE_FAIL2BAN=1`:
    # cap_add:
    #   - NET_ADMI
    #N

    healthcheck:
      test: "ss --listening --tcp | grep -P 'LISTEN.+:smtp' || exit 1"
      timeout: 3s
      retries: 0

```

Before start your docker, create `/etc/postfix/main.cf` and you are good to go, this `main.cf` use IPv4 as your Gophish identity. Make sure to serve your SMTP in 25, or mapping it to another services, since we are running non-encryption SMTP.

```
myhostname = example.com
mydomain = example.com
myorigin = $mydomain
inet_interfaces = all
inet_protocols = ipv4

mynetworks = 127.0.0.0/8, {Your gophish Ipv4}


smtpd_use_tls = yes
smtpd_tls_cert_file = /etc/letsencrypt/live/example.com/fullchain.pem
smtpd_tls_key_file = /etc/letsencrypt/live/example.com/privkey.pem
smtpd_tls_security_level = may
smtpd_tls_protocols = !SSLv2, !SSLv3
smtpd_tls_session_cache_database = btree:${data_directory}/smtpd_scache
smtp_tls_CAfile = /etc/letsencrypt/live/example.com/chain.pem

smtpd_sasl_auth_enable = yes
smtpd_sasl_security_options = noanonymous
smtpd_sasl_local_domain = $myhostname


smtpd_recipient_restrictions =
    permit_mynetworks,
    permit_sasl_authenticated,
    reject_unauth_destination


mydestination = $myhostname, localhost.$mydomain, localhost, $mydomain
relay_domains =
home_mailbox = Maildir/
alias_maps = hash:/etc/aliases
alias_database = hash:/etc/aliases
compatibility_level = 2

smtpd_banner = $myhostname ESMTP $mail_name (Ubuntu)


```
