# README – Datasets

## Datei: `synth_labs_1000_cases.csv`

**Enthaltene Spalten (Auszug):**

`case_id`, `patient_id`, `sex`, `age_years`, `specimen_datetime`

**Laborwerte (Auswahl):**

`Natrium`, `Kalium`, `Kreatinin`, `eGFR`, `Glukose`, `Hb`, `Leukozyten`, `Thrombozyten`, `CRP`, `ALT`, `AST`, `Bilirubin`, `Albumin`, `INR`, `Laktat`

Zu jedem Laborwert existieren zusätzliche Felder:

 `*_flag`, `*_ref_low`, `*_ref_high`

**Quelle:**  
synthetisch generiert (keine echten Patientendaten)


# Datei: `synthetic_device_motion_fall_data.csv`

## Struktur

- `patient_id`
- `timestamp` (stündliche Aggregation)
- `movement_index_0_100` (Bewegungsintensität)
- `micro_movements_count` (Lagewechsel / Mikrobewegungen)
- `bed_exit_detected_0_1`
- `fall_event_0_1`
- `impact_magnitude_g` (nur bei Sturz befüllt)
- `post_fall_immobility_minutes` (nur bei Sturz befüllt)

## Umfang

- 200 Patienten  
- 5 Tage Monitoring  
- 24 Datensätze pro Tag  

→ **24.000 Zeitreihen-Einträge**

**Quelle:**  
synthetisch generiert (keine echten Patientendaten)

# Datei: `synthetic_device_raw_1hz_motion_fall.csv`

## Umfang

- 5 Patienten  
- 6 Stunden Monitoring  
- 1 Messung pro Sekunde  

→ **108.000 Zeilen**

## Spalten (Auszug)

### Identifikation

- `patient_id`
- `device_id`
- `timestamp`

### Zustand / Bewegung

- `bed_occupied_0_1`
- `movement_score_0_100`

### IMU / Bewegungssignal

- `accel_x_m_s2`
- `accel_y_m_s2`
- `accel_z_m_s2`
- `accel_magnitude_g`

### Druckzonen Matratze

- `pressure_zone1_0_100`
- `pressure_zone2_0_100`
- `pressure_zone3_0_100`
- `pressure_zone4_0_100`

### Events

- `bed_exit_event_0_1`
- `bed_return_event_0_1`
- `fall_event_0_1`
- `impact_magnitude_g`
- `event_id`

## Hinweise zur Logik (synthetisch)

**Bed-Exit**

- erzeugt kurze Bewegungs-Bursts  
- `bed_occupied_0_1` wird während der „away“-Phase auf `0` gesetzt

**Fall**

- erzeugt einen Spike (`impact_magnitude_g` + `event_id`)
- anschließend Phase sehr niedriger Bewegung (**Immobility**)


# Datei: `synthetic_medication_raw_inpatient.csv`

## Inhalt (eine Datei mit 3 Record-Typen)

### `record_type = ORDER`

Verordnung:

- Start / Stop
- Frequenz
- Dosis
- Route
- PRN

### `record_type = CHANGE`

Dosisanpassung während des Verlaufs:

- gleiche `order_id`
- neues `order_start_datetime`

### `record_type = ADMIN`

Einzelne Medikamentengaben:

- Status: `given`, `missed`, `held`, `refused`
- optionale Notiz

## Umfang

- **200 stationäre Aufenthalte**
- **14.553 Zeilen** (Orders + Admins + Changes)

## Wichtige Spalten (Auszug)

### Aufenthalt

- `patient_id`
- `encounter_id`
- `ward`
- `admission_datetime`
- `discharge_datetime`

### Verordnung

- `order_id`
- `order_uuid`
- `medication_code_atc`
- `medication_name`
- `route`
- `dose`
- `dose_unit`
- `frequency`
- `order_start_datetime`
- `order_stop_datetime`
- `is_prn_0_1`
- `indication`
- `prescriber_role`
- `order_status`

### Gabe

- `administration_datetime`
- `administered_dose`
- `administered_unit`
- `administration_status`
- `note`

**Quelle:**  
synthetisch generiert (keine echten Patientendaten)

# Datei: `synthetic_nursing_daily_reports.csv`

## Umfang

- **50 Fälle**
- **2–5 Tage pro Fall**
- **1 Pflegebericht pro Tag**

→ **181 Einträge**

## Struktur

- `case_id`
- `patient_id`
- `ward`
- `report_date`
- `shift`
- `nursing_note_free_text` (unstrukturierter Freitext)

## Inhalt der Texte

Typische pflegerische Inhalte:

- Zustandsbeschreibung
- Symptome / Beobachtungen
- Maßnahmen
- Evaluation


# EPA Data

## Gültige Fälle

Nur Fälle mit folgenden Informationen:

- Einschätzungstyp
- Einschätzungsdatum
- FallNr (ID)

IID-Beispiele: `E0I001`, `E2I225`, `E2I222`


## Datei: `epaAC-Data-1.csv`

Tabellarische Struktur:

- Pro **IID ein Datensatz**
- **Duplikate möglich**
- Maßgeblich ist **der letzte Datensatz** bei Duplikaten


## Datei: `epaAC-Data-2.csv`

- **Pro Einschätzung ein Datensatz**
- Headerinformationen beziehen sich auf **SID**


## Datei: `epaAC-Data-3.csv`

- **Pro Einschätzung ein Datensatz**
- Headerinformationen siehe: `IID-SID-ITEM.csv`


## Datei: `epaAC-Data-4.xlsx`

- **Pro Einschätzung ein Datensatz**
- Headerinformationen siehe: `IID-SID-ITEM.csv`

## Datei: `epaAC-Data-5.csv`

- **Pro Einschätzung ein Datensatz**
- Headerinformationen verschlüsselt

# Datei: `synthetic_cases_icd10_ops.csv`

## Struktur

- **50 Fälle**
- Bezug zu zuvor erzeugten `case_id` / `patient_id`

## Felder

- `primary_icd10_code`
- `primary_icd10_description_en`
- `secondary_icd10_codes`
- `secondary_icd10_descriptions_en`
- `ops_codes`
- `ops_descriptions_en`

Zusätzliche Informationen:

- Station
- Aufnahmedatum
- Entlassdatum
- Verweildauer


# Fehlende Werte

## Pflichtfelder

Folgende Felder **müssen vorhanden sein**:

- `case_id`
- `patient_id`

Falls diese fehlen → **Zeile wird entfernt**


## Format von `case_id`

Kann in verschiedenen Schreibweisen vorkommen:
- `CASE-0135`
- `0135`
- `135`

Datentyp:

- **Integer**

## Werte, die als fehlend gelten

Folgende Inhalte werden als **NULL** interpretiert:

- `NULL`
- `Missing`
- `unknow`
- `NaN`
- `N/A`
- nur Leerzeichen

Ausnahme: `case_id`, `patient_id`

Wenn diese fehlen → **Datensatz wird entfernt**
