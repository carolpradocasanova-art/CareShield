-- Caregiver soft-delete migration
-- Run manually in the Supabase SQL editor when ready.
--
-- Goal: hide removed caregivers from "Who's logged in?" while keeping their row
-- so documents.medications/conditions can still show "Added by {name}" via
-- uploaded_by_caregiver_id (or similar FKs).

-- Option A (recommended): nullable deleted_at timestamp
alter table public.caregivers
  add column if not exists deleted_at timestamptz;

comment on column public.caregivers.deleted_at is
  'When set, caregiver is hidden from the login selector but remains linkable for attribution.';

create index if not exists caregivers_active_idx
  on public.caregivers (deleted_at)
  where deleted_at is null;

-- Option B (alternative): boolean flag instead of deleted_at
-- alter table public.caregivers
--   add column if not exists is_deleted boolean not null default false;
-- create index if not exists caregivers_is_deleted_idx
--   on public.caregivers (is_deleted)
--   where is_deleted = false;

-- Preserve attribution if a caregiver row is ever hard-deleted by mistake:
-- change ON DELETE CASCADE / RESTRICT on child tables to SET NULL instead.
-- Example for documents (adjust constraint name if different in your project):
--
-- alter table public.documents
--   drop constraint if exists documents_uploaded_by_caregiver_id_fkey;
-- alter table public.documents
--   add constraint documents_uploaded_by_caregiver_id_fkey
--   foreign key (uploaded_by_caregiver_id)
--   references public.caregivers(id)
--   on delete set null;

-- Active caregivers only (for selector queries):
-- select * from public.caregivers where deleted_at is null order by created_at;
--
-- Attribution lookup (include soft-deleted):
-- select display_name from public.caregivers where id = :caregiver_id;
