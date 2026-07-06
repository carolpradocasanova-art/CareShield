-- =========================================================
-- CareShield: patients, medications, and conditions schema
-- =========================================================
-- This replaces "saved in the browser only" with real,
-- persistent data in Supabase, shared across all caregivers.
 
-- ---------------------------------------------------------
-- 1. Caregivers (the C / M / N buttons shown under
--    "Who's logged in?"). Simple, no real auth yet.
-- ---------------------------------------------------------
create table if not exists caregivers (
  id bigint primary key generated always as identity,
  initial text not null,        -- e.g. 'C', 'M', 'N'
  display_name text not null,   -- e.g. 'Carlos (son)'
  created_at timestamptz default now()
);
 
-- ---------------------------------------------------------
-- 2. Patients (the P / M buttons shown under
--    "Currently caring for"). This is what's missing today.
-- ---------------------------------------------------------
create table if not exists patients (
  id bigint primary key generated always as identity,
  initial text not null,        -- e.g. 'P', 'M'
  display_name text not null,   -- e.g. 'Mum', 'Dad'
  created_at timestamptz default now()
);
 
-- ---------------------------------------------------------
-- 3. Medications stored per patient
--    (what shows up under "STORED MEDICATIONS")
-- ---------------------------------------------------------
create table if not exists medications (
  id bigint primary key generated always as identity,
  patient_id bigint not null references patients(id) on delete cascade,
  name text not null,           -- e.g. 'Paracetamol 500 mg'
  dosage_instructions text,     -- e.g. 'As needed', 'Once daily'
  source_document_id bigint,    -- optional reference to the uploaded document
  created_at timestamptz default now()
);
 
create index if not exists idx_medications_patient_id on medications(patient_id);
 
-- ---------------------------------------------------------
-- 4. Conditions stored per patient
--    (what shows up under "STORED CONDITIONS", empty in your screenshot)
-- ---------------------------------------------------------
create table if not exists conditions (
  id bigint primary key generated always as identity,
  patient_id bigint not null references patients(id) on delete cascade,
  name text not null,           -- e.g. 'Hypertension'
  notes text,
  source_document_id bigint,
  created_at timestamptz default now()
);
 
create index if not exists idx_conditions_patient_id on conditions(patient_id);
 
-- ---------------------------------------------------------
-- 5. Uploaded documents (discharge plan, prescription, etc.)
--    Replaces/extends your current "patient_plan" table,
--    but now linked to a specific patient.
-- ---------------------------------------------------------
create table if not exists documents (
  id bigint primary key generated always as identity,
  patient_id bigint not null references patients(id) on delete cascade,
  file_name text,
  raw_text text,                -- text extracted from the PDF
  uploaded_by_caregiver_id bigint references caregivers(id),
  created_at timestamptz default now()
);
 
create index if not exists idx_documents_patient_id on documents(patient_id);
 
-- ---------------------------------------------------------
-- 6. Enable RLS and keep access open for now
--    (same as your current "Allow all access for now" policy)
-- ---------------------------------------------------------
alter table caregivers enable row level security;
alter table patients enable row level security;
alter table medications enable row level security;
alter table conditions enable row level security;
alter table documents enable row level security;
 
create policy "Allow all access for now" on caregivers for all using (true) with check (true);
create policy "Allow all access for now" on patients for all using (true) with check (true);
create policy "Allow all access for now" on medications for all using (true) with check (true);
create policy "Allow all access for now" on conditions for all using (true) with check (true);
create policy "Allow all access for now" on documents for all using (true) with check (true);
 
-- ---------------------------------------------------------
-- 7. Seed data: the caregivers and patients already visible
--    on screen (Carlos/M/N and Dad/Mum).
--    ADJUST these names if they don't match the real ones.
-- ---------------------------------------------------------
insert into caregivers (initial, display_name) values
  ('C', 'Carlos (son)'),
  ('M', 'M'),
  ('N', 'N');
 
insert into patients (initial, display_name) values
  ('P', 'Dad'),
  ('M', 'Mum');
 
-- IMPORTANT: we deliberately do not insert any medications or
-- conditions for these patients. They should stay empty until
-- a real document is uploaded and processed. If your frontend
-- still shows "Paracetamol 500mg" for a brand-new patient
-- without ever querying this table, that's the bug to fix in
-- the app code (see note below).
 





 create table medication_references (
    id bigserial primary key,
    medication_name text not null,
    image_b64 text not null,
    description text,
    created_at timestamp with time zone default now()
);




CREATE TABLE IF NOT EXISTS medication_logs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  patient_id TEXT NOT NULL,
  medication_name TEXT NOT NULL,
  scheduled_time TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('taken', 'missed')),
  logged_at TIMESTAMPTZ DEFAULT now(),
  caregiver_id TEXT,
  notes TEXT
);

CREATE INDEX idx_medication_logs_patient_id ON medication_logs(patient_id);
CREATE INDEX idx_medication_logs_logged_at ON medication_logs(logged_at);





create or replace function match_medical_knowledge(
    query_embedding vector(1536),
    match_count int
)
returns table (
    id bigint,
    title text,
    content text,
    source text,
    source_url text,
    similarity float
)
language sql stable
as $$
    select
        id, title, content, source, source_url,
        1 - (embedding <=> query_embedding) as similarity
    from medical_knowledge
    order by embedding <=> query_embedding
    limit match_count;
$$;





-- Enable the pgvector extension
create extension if not exists vector;

-- Create the table to hold your medical knowledge chunks
create table medical_knowledge (
    id bigserial primary key,
    source text not null,
    source_url text,
    title text,
    content text not null,
    embedding vector(1536)
);

-- Create an index for fast similarity search
create index on medical_knowledge using ivfflat (embedding vector_cosine_ops);




create table patient_plan (
  id bigint primary key generated always as identity,
  created_at timestamptz default now(),
  raw_text text,
  medications text
);

alter table patient_plan enable row level security;

create policy "Allow all access for now"
on patient_plan
for all
using (true)
with check (true);




CREATE TABLE IF NOT EXISTS medication_logs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  patient_id TEXT NOT NULL,
  medication_name TEXT NOT NULL,
  scheduled_time TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('taken', 'missed')),
  logged_at TIMESTAMPTZ DEFAULT now(),
  caregiver_id TEXT,
  notes TEXT
);

CREATE INDEX idx_medication_logs_patient_id ON medication_logs(patient_id);
CREATE INDEX idx_medication_logs_logged_at ON medication_logs(logged_at);





create table medication_references (
    id bigserial primary key,
    medication_name text not null,
    image_b64 text not null,
    description text,
    created_at timestamp with time zone default now()
);





create or replace function match_medical_knowledge(
    query_embedding vector(1536),
    match_count int
)
returns table (
    id bigint,
    title text,
    content text,
    source text,
    source_url text,
    similarity float
)
language sql stable
as $$
    select
        id, title, content, source, source_url,
        1 - (embedding <=> query_embedding) as similarity
    from medical_knowledge
    order by embedding <=> query_embedding
    limit match_count;
$$;





-- Enable the pgvector extension
create extension if not exists vector;

-- Create the table to hold your medical knowledge chunks
create table medical_knowledge (
    id bigserial primary key,
    source text not null,
    source_url text,
    title text,
    content text not null,
    embedding vector(1536)
);

-- Create an index for fast similarity search
create index on medical_knowledge using ivfflat (embedding vector_cosine_ops);




create table patient_plan (
  id bigint primary key generated always as identity,
  created_at timestamptz default now(),
  raw_text text,
  medications text
);

alter table patient_plan enable row level security;

create policy "Allow all access for now"
on patient_plan
for all
using (true)
with check (true);