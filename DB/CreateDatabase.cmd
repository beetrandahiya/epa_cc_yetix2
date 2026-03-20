@echo off
rem -- example for server name = <ComputerName>\<InstanceName>
rem -- default is local machine and default instance
rem -- command line utility sqlcmd.exe must be installed and accessible
set ServerName=[Instance\Name]
set DatabaseName=Hack2026
set AnmeldeID=sa
set Kennwort=[Password]

echo Anlegen der Datenbank
sqlcmd -S %ServerName% -U %AnmeldeID% -P %Kennwort% -Q "CREATE DATABASE %DatabaseName%"

echo Anlegen der Tabellee
sqlcmd -S %ServerName% -U %AnmeldeID% -P %Kennwort% /d %DatabaseName% -i CreateimportTables.sql
