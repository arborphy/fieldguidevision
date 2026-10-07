-- Shared anonymous review for the 182-image photographic-scale audit.
-- Anonymous Supabase users still receive auth.uid() and the authenticated role.

create table if not exists public.scale_review_batches (
  id uuid primary key,
  slug text unique not null,
  title text not null,
  status text not null default 'open' check (status in ('open', 'closed')),
  created_at timestamptz not null default now()
);

create table if not exists public.scale_review_items (
  batch_id uuid not null references public.scale_review_batches(id) on delete cascade,
  image_id text not null,
  display_order integer not null,
  species text not null,
  organ_category text not null,
  subview text not null,
  provisional_scale text not null,
  provisional_confidence text not null,
  primary key (batch_id, image_id),
  unique (batch_id, display_order)
);

create table if not exists public.scale_annotations (
  batch_id uuid not null,
  image_id text not null,
  reviewer_id uuid not null default auth.uid() references auth.users(id),
  human_scale text not null check (human_scale in ('distant', 'mid-range', 'close-up', 'uncertain')),
  confidence text not null check (confidence in ('high', 'medium', 'low')),
  note text,
  revision integer not null default 1,
  submitted_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  primary key (batch_id, image_id, reviewer_id),
  foreign key (batch_id, image_id)
    references public.scale_review_items(batch_id, image_id) on delete cascade
);

create table if not exists public.scale_annotation_events (
  id bigint generated always as identity primary key,
  batch_id uuid not null,
  image_id text not null,
  reviewer_id uuid not null references auth.users(id),
  payload jsonb not null,
  created_at timestamptz not null default now()
);

alter table public.scale_review_batches enable row level security;
alter table public.scale_review_items enable row level security;
alter table public.scale_annotations enable row level security;
alter table public.scale_annotation_events enable row level security;

drop policy if exists "scale batches readable" on public.scale_review_batches;
create policy "scale batches readable" on public.scale_review_batches
  for select to authenticated using (true);

drop policy if exists "scale items readable" on public.scale_review_items;
create policy "scale items readable" on public.scale_review_items
  for select to authenticated using (true);

drop policy if exists "reviewers read own scale annotations" on public.scale_annotations;
create policy "reviewers read own scale annotations" on public.scale_annotations
  for select to authenticated using (reviewer_id = auth.uid());

drop policy if exists "reviewers insert own scale annotations" on public.scale_annotations;
create policy "reviewers insert own scale annotations" on public.scale_annotations
  for insert to authenticated with check (reviewer_id = auth.uid());

drop policy if exists "reviewers update own scale annotations" on public.scale_annotations;
create policy "reviewers update own scale annotations" on public.scale_annotations
  for update to authenticated using (reviewer_id = auth.uid()) with check (reviewer_id = auth.uid());

drop policy if exists "reviewers read own scale events" on public.scale_annotation_events;
create policy "reviewers read own scale events" on public.scale_annotation_events
  for select to authenticated using (reviewer_id = auth.uid());

drop policy if exists "reviewers insert own scale events" on public.scale_annotation_events;
create policy "reviewers insert own scale events" on public.scale_annotation_events
  for insert to authenticated with check (reviewer_id = auth.uid());

grant select on public.scale_review_batches, public.scale_review_items to authenticated;
grant select, insert, update on public.scale_annotations to authenticated;
grant select, insert on public.scale_annotation_events to authenticated;
revoke delete on public.scale_annotations, public.scale_annotation_events from anon, authenticated;

create or replace function public.submit_scale_annotation(
  p_batch_id uuid,
  p_image_id text,
  p_human_scale text,
  p_confidence text,
  p_note text default null
) returns public.scale_annotations
language plpgsql
security invoker
set search_path = public
as $$
declare
  result public.scale_annotations;
begin
  if auth.uid() is null then
    raise exception 'Authentication required';
  end if;

  insert into public.scale_annotations (
    batch_id, image_id, reviewer_id, human_scale, confidence, note
  ) values (
    p_batch_id, p_image_id, auth.uid(), p_human_scale, p_confidence, nullif(trim(p_note), '')
  )
  on conflict (batch_id, image_id, reviewer_id) do update set
    human_scale = excluded.human_scale,
    confidence = excluded.confidence,
    note = excluded.note,
    revision = public.scale_annotations.revision + 1,
    updated_at = now()
  returning * into result;

  insert into public.scale_annotation_events (batch_id, image_id, reviewer_id, payload)
  values (p_batch_id, p_image_id, auth.uid(), to_jsonb(result));

  return result;
end;
$$;

revoke all on function public.submit_scale_annotation(uuid, text, text, text, text) from public, anon;
grant execute on function public.submit_scale_annotation(uuid, text, text, text, text) to authenticated;

insert into public.scale_review_batches (id, slug, title)
values ('00000000-0000-4000-8000-000000000002', 'scale-audit-v1', 'BioImages photographic-scale audit')
on conflict (id) do update set title = excluded.title;

-- Preserve annotations while freeing the unique display-order range for an expanded batch.
update public.scale_review_items
set display_order = display_order + 10000
where batch_id = '00000000-0000-4000-8000-000000000002';

insert into public.scale_review_items (
  batch_id, image_id, display_order, species, organ_category, subview,
  provisional_scale, provisional_confidence
) values
('00000000-0000-4000-8000-000000000002', 'baskauf/10975', 1, 'Magnolia acuminata', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11614', 2, 'Cornus florida', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11695', 3, 'Lindera benzoin', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/12039', 4, 'Carpinus caroliniana', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15203', 5, 'Juglans nigra', 'twig', 'close-up winter leaf scar/bud', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15339', 6, 'Populus deltoides', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15503', 7, 'Hamamelis virginiana', 'fruit', 'lateral or general close-up', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/20454', 8, 'Toxicodendron radicans', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/25646', 9, 'Fraxinus americana', 'fruit', 'lateral or general close-up', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/30758', 10, 'Cornus amomum', 'twig', 'close-up winter leaf scar/bud', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/31641', 11, 'Acer negundo', 'inflorescence', 'lateral view of flower', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/35218', 12, 'Cornus amomum', 'inflorescence', 'lateral view of flower', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/38140', 13, 'Betula lenta', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/39171', 14, 'Ailanthus altissima', 'twig', 'close-up winter terminal bud', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/39176', 15, 'Fagus grandifolia', 'twig', 'close-up winter terminal bud', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/49248', 16, 'Betula alleghaniensis', 'leaf', 'margin of upper + lower surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/52472', 17, 'Quercus palustris', 'leaf', 'margin of upper + lower surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/52662', 18, 'Rhus aromatica', 'leaf', 'margin of upper + lower surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/54962', 19, 'Acer negundo', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/66187', 20, 'Gleditsia triacanthos', 'inflorescence', 'lateral view of flower', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/b5068', 21, 'Carpinus caroliniana', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2003', 22, 'Fagus grandifolia', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2014', 23, 'Quercus palustris', 'leaf', 'margin of upper + lower surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2063', 24, 'Fagus grandifolia', 'leaf', 'whole upper surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2356', 25, 'Quercus palustris', 'leaf', 'margin of upper + lower surface', 'close-up', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10419', 26, 'Rhus copallinum', 'fruit', 'lateral or general close-up', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10574', 27, 'Nyssa sylvatica', 'inflorescence', 'whole - unspecified', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11174', 28, 'Ginkgo biloba', 'leaf', 'showing orientation on twig', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11454', 29, 'Robinia pseudoacacia', 'bark', 'of a large tree', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11645', 30, 'Tilia americana', 'leaf', 'showing orientation on twig', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11902', 31, 'Gymnocladus dioicus', 'bark', 'of a small tree or small branch', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11958', 32, 'Ginkgo biloba', 'bark', 'of a large tree', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/12438', 33, 'Quercus palustris', 'bark', 'of a large tree', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/12467', 34, 'Fraxinus americana', 'leaf', 'unspecified', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/12516', 35, 'Quercus alba', 'leaf', 'unspecified', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15734', 36, 'Cephalanthus occidentalis', 'fruit', 'as borne on the plant', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/17455', 37, 'Fraxinus pennsylvanica', 'twig', 'winter overall', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/17544', 38, 'Pinus strobus', 'bark', 'of a medium tree or large branch', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/26143', 39, 'Sambucus racemosa', 'leaf', 'whole upper surface', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/29699', 40, 'Juglans nigra', 'twig', 'unspecified', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/32629', 41, 'Pyrus calleryana', 'leaf', 'showing orientation on twig', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/35716', 42, 'Zanthoxylum americanum', 'leaf', 'showing orientation on twig', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/55799', 43, 'Hamamelis virginiana', 'leaf', 'showing orientation on twig', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/ac1449', 44, 'Quercus alba', 'bark', 'of a large tree', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em1993', 45, 'Fagus grandifolia', 'twig', 'orientation of petioles', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2095', 46, 'Quercus alba', 'twig', 'orientation of petioles', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2146', 47, 'Quercus bicolor', 'leaf', 'showing orientation on twig', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2319', 48, 'Quercus palustris', 'twig', 'orientation of petioles', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'ssmv/2-886-02', 49, 'Acer saccharum', 'bark', 'of a large tree', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'thomas/0049-08-02', 50, 'Cercis canadensis', 'inflorescence', 'whole - unspecified', 'close-up', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10341', 51, 'Ulmus americana', 'whole tree (or vine)', 'winter', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10643', 52, 'Prunus serotina', 'whole tree (or vine)', 'view up trunk', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11322', 53, 'Sassafras albidum', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11497', 54, 'Tsuga canadensis', 'whole tree', 'view up trunk', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11930', 55, 'Acer rubrum', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/12321', 56, 'Quercus muehlenbergii', 'whole tree (or vine)', 'view up trunk', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/35191', 57, 'Quercus alba', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/65657', 58, 'Dirca palustris', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/89356', 59, 'Juglans nigra', 'whole tree (or vine)', 'winter', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/89384', 60, 'Juniperus virginiana', 'whole tree (or vine)', 'winter', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/89403', 61, 'Acer negundo', 'whole tree (or vine)', 'winter', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90661', 62, 'Quercus macrocarpa', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90696', 63, 'Quercus rubra', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90750', 64, 'Juglans nigra', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90807', 65, 'Ulmus americana', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90826a', 66, 'Fraxinus americana', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90872', 67, 'Platanus occidentalis', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90899', 68, 'Fraxinus americana', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90911', 69, 'Platanus occidentalis', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/90912', 70, 'Quercus velutina', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/91091', 71, 'Fraxinus pennsylvanica', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/91127', 72, 'Juglans nigra', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/b5063', 73, 'Carpinus caroliniana', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'ssmv/2-638-02', 74, 'Fraxinus americana', 'whole tree (or vine)', 'general', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'ssmv/2-886-03', 75, 'Acer saccharum', 'whole tree (or vine)', 'view up trunk', 'distant', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/00000', 76, 'Acer rubrum', 'inflorescence', 'whole - unspecified', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10225', 77, 'Pinus strobus', 'cone', 'female - mature open', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10464', 78, 'Platanus occidentalis', 'fruit', 'as borne on the plant', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10897', 79, 'Acer negundo', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10904', 80, 'Ailanthus altissima', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11054', 81, 'Populus deltoides', 'fruit', 'as borne on the plant', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11086', 82, 'Lindera benzoin', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11534', 83, 'Castanea dentata', 'bark', 'of a large tree', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11732', 84, 'Yucca filamentosa', 'leaf', 'showing orientation on twig', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/13165', 85, 'Ailanthus altissima', 'fruit', 'as borne on the plant', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/13516', 86, 'Rhus glabra', 'fruit', 'as borne on the plant', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/13534', 87, 'Rhus copallinum', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15728', 88, 'Carpinus caroliniana', 'bark', 'of a small tree or small branch', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16741', 89, 'Acer saccharum', 'leaf', 'showing orientation on twig', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/19579', 90, 'Ulmus rubra', 'fruit', 'as borne on the plant', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/26114', 91, 'Sambucus racemosa', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/30778', 92, 'Viburnum opulus', 'whole tree (or vine)', 'winter', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/32255', 93, 'Rhus aromatica', 'whole tree (or vine)', 'winter', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/35713', 94, 'Zanthoxylum americanum', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/43402', 95, 'Acer nigrum', 'whole tree (or vine)', 'view up trunk', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/43418', 96, 'Acer nigrum', 'leaf', 'showing orientation on twig', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/43553', 97, 'Thuja occidentalis', 'whole tree', 'view up trunk', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/50713', 98, 'Acer saccharum', 'inflorescence', 'whole - unspecified', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/50749', 99, 'Acer rubrum', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/65662', 100, 'Dirca palustris', 'whole tree (or vine)', 'general', 'distant', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10207', 101, 'Cercis canadensis', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10414', 102, 'Rhus glabra', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11288', 103, 'Acer saccharum', 'leaf', 'showing orientation on twig', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/12700', 104, 'Rhus aromatica', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/13075', 105, 'Quercus velutina', 'leaf', 'showing orientation on twig', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/13127', 106, 'Quercus rubra', 'leaf', 'showing orientation on twig', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15007', 107, 'Sorbus americana', 'leaf', 'showing orientation on twig', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15687', 108, 'Sambucus nigra', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15935', 109, 'Juglans nigra', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15951', 110, 'Toxicodendron radicans', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16835', 111, 'Platanus occidentalis', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/2014-08-23-16-40-31', 112, 'Fraxinus quadrangulata', 'leaf', 'showing orientation on twig', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/21602', 113, 'Gleditsia triacanthos', 'leaf', 'showing orientation on twig', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/24270', 114, 'Gymnocladus dioicus', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/28499', 115, 'Acer negundo', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/35296', 116, 'Prunus virginiana', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/38439', 117, 'Gleditsia triacanthos', 'twig', 'orientation of petioles', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/50392', 118, 'Picea glauca', 'bark', 'of a medium tree or large branch', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/d0426', 119, 'Celtis occidentalis', 'leaf', 'showing orientation on twig', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'hessd/e5353', 120, 'Rhus aromatica', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/b5071', 121, 'Carpinus caroliniana', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/b5095', 122, 'Cornus florida', 'fruit', 'as borne on the plant', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em1998', 123, 'Fagus grandifolia', 'twig', 'orientation of petioles', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2316', 124, 'Quercus palustris', 'leaf', 'showing orientation on twig', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'ssmv/2-185-04', 125, 'Ginkgo biloba', 'leaf', 'showing orientation on twig', 'mid-range', 'high'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10503', 126, 'Fagus grandifolia', 'inflorescence', 'whole - male', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10675', 127, 'Aesculus hippocastanum', 'inflorescence', 'whole - unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/12560', 128, 'Rhus glabra', 'inflorescence', 'whole - unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/13537', 129, 'Rhus copallinum', 'inflorescence', 'whole - unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/15075', 130, 'Carpinus caroliniana', 'inflorescence', 'whole - male', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16046', 131, 'Acer negundo', 'twig', 'unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/17534', 132, 'Ailanthus altissima', 'twig', 'winter overall', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/18748', 133, 'Ostrya virginiana', 'inflorescence', 'whole - male', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/18850', 134, 'Quercus macrocarpa', 'inflorescence', 'whole - unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/19171', 135, 'Carpinus caroliniana', 'inflorescence', 'whole - unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/21454', 136, 'Salix nigra', 'inflorescence', 'whole - male', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/23022', 137, 'Juglans nigra', 'inflorescence', 'whole - unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/31283', 138, 'Pyrus calleryana', 'twig', 'winter overall', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/34216', 139, 'Pinus strobus', 'cone', 'male', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/49326', 140, 'Sorbus americana', 'inflorescence', 'whole - unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/50913', 141, 'Gymnocladus dioicus', 'bark', 'of a large tree', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/51172', 142, 'Fraxinus pennsylvanica', 'inflorescence', 'whole - female', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/51406', 143, 'Nyssa sylvatica', 'inflorescence', 'whole - female', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/51572', 144, 'Carya cordiformis', 'inflorescence', 'whole - unspecified', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/66138', 145, 'Gleditsia triacanthos', 'inflorescence', 'whole - female', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/66166', 146, 'Gleditsia triacanthos', 'fruit', 'immature', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/66168', 147, 'Gleditsia triacanthos', 'fruit', 'immature', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/66183', 148, 'Gleditsia triacanthos', 'inflorescence', 'whole - male', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/66184', 149, 'Gleditsia triacanthos', 'inflorescence', 'whole - male', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2413', 150, 'Quercus rubra', 'bark', 'of a large tree', 'mid-range', 'medium'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11464', 151, 'Tilia americana', 'leaf', 'showing orientation on twig', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11899', 152, 'Gymnocladus dioicus', 'leaf', 'unspecified', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11945', 153, 'Aesculus hippocastanum', 'whole tree (or vine)', 'general', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16712', 154, 'Nyssa sylvatica', 'whole tree (or vine)', 'view up trunk', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16720', 155, 'Fagus grandifolia', 'leaf', 'showing orientation on twig', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/24248', 156, 'Gymnocladus dioicus', 'leaf', 'whole upper surface', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/26495', 157, 'Viburnum lantanoides', 'whole tree (or vine)', 'general', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/31257', 158, 'Pyrus calleryana', 'whole tree (or vine)', 'winter', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/32045', 159, 'Pyrus calleryana', 'whole tree (or vine)', 'general', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/32863', 160, 'Fraxinus quadrangulata', 'whole tree (or vine)', 'general', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'kaufmannm/ke062', 161, 'Populus tremuloides', 'whole tree (or vine)', 'general', 'uncertain', 'conflict'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10462', 162, 'Platanus occidentalis', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/10499', 163, 'Quercus rubra', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11251', 164, 'Quercus alba', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11449', 165, 'Betula alleghaniensis', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/11458', 166, 'Robinia pseudoacacia', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/13650', 167, 'Carya ovata', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/13843', 168, 'Carya laciniosa', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16502', 169, 'Juglans cinerea', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/16842', 170, 'Quercus alba', 'seed', 'unspecified', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/20950', 171, 'Ulmus americana', 'fruit', 'unspecified', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/26421', 172, 'Betula alleghaniensis', 'inflorescence', 'whole - male', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/27896', 173, 'Carya cordiformis', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/27897', 174, 'Carya cordiformis', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/50500', 175, 'Pinus resinosa', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/79653', 176, 'Quercus macrocarpa', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'baskauf/89955', 177, 'Platanus occidentalis', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'bassettst/sb569', 178, 'Betula papyrifera', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/ac1448', 179, 'Quercus alba', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/ac1469', 180, 'Quercus bicolor', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2106', 181, 'Quercus alba', 'bark', 'of a large tree', 'uncertain', 'low'),
('00000000-0000-4000-8000-000000000002', 'kirchoff/em2192', 182, 'Fagus grandifolia', 'bark', 'of a large tree', 'uncertain', 'low')
on conflict (batch_id, image_id) do update set
  display_order = excluded.display_order,
  species = excluded.species,
  organ_category = excluded.organ_category,
  subview = excluded.subview,
  provisional_scale = excluded.provisional_scale,
  provisional_confidence = excluded.provisional_confidence;
