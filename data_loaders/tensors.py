import torch

def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def n_joints_to_mask(n_joints, max_joints):
    mask = torch.arange(max_joints + 1, device=n_joints.device).expand(len(n_joints), max_joints + 1) < (n_joints.unsqueeze(1) + 1)
    mask = mask.unsqueeze(2).float() * mask.unsqueeze(1).float()
    return mask

def length_to_temp_mask(max_len_mask, lengths, max_len):
    mask = torch.arange(max_len + 1, device=lengths.device).expand(len(lengths), max_len + 1) < (lengths.unsqueeze(1) + 1)
    mask = mask.unsqueeze(2).float() * mask.unsqueeze(1).float()
    mask = mask.logical_and(max_len_mask)
    return mask

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def create_padded_relation(relation_np, max_joints, n_joints):
    relation = torch.from_numpy(relation_np)
    padded_relation = torch.zeros((max_joints, max_joints))
    padded_relation[:n_joints, :n_joints] = relation
    return padded_relation

def truebones_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    tposfirstframebatch = [b['tpos_first_frame'] for b in notnone_batches]
    meanbatch = [b['mean'] for b in notnone_batches]
    stdbatch = [b['std'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]
    if 'n_joints' in notnone_batches[0]:
        jointsnumbatch = [b['n_joints'] for b in notnone_batches]
    else:
        jointsnumbatch = [22 for b in notnone_batches]

    if 'temporal_mask' in notnone_batches[0]:
        temporalmasksbatch = [b['temporal_mask'] for b in notnone_batches]
    if 'crop_start_ind' in notnone_batches[0]:
        cropstartindbatch = [b['crop_start_ind'] for b in notnone_batches]

    if 'video_dino' in notnone_batches[0]:
        videodinobatch = [b['video_dino'] for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    tposfirstframebatchTensor = collate_tensors(tposfirstframebatch)
    meanbatchTensor = collate_tensors(meanbatch)
    stdbatchTensor = collate_tensors(stdbatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    cropstartindTensor = torch.as_tensor(cropstartindbatch)
    lengthsmaskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)
    jointsnumbatchTensor = torch.as_tensor(jointsnumbatch)
    jointsmaskbatchTensor = n_joints_to_mask(jointsnumbatchTensor, databatchTensor.shape[1]).unsqueeze(1).unsqueeze(1)

    collated_temporalmasksbatch = collate_tensors(temporalmasksbatch)
    maskbatchTensor = length_to_temp_mask(collated_temporalmasksbatch, lenbatchTensor, collated_temporalmasksbatch[0].size(0) - 1).unsqueeze(1).unsqueeze(1)

    videodinobatchTensor = collate_tensors(videodinobatch)

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor, 'lengths_mask': lengthsmaskbatchTensor, 'tpos_first_frame': tposfirstframebatchTensor, 'crop_start_ind': cropstartindTensor, 'mean': meanbatchTensor, 'std':stdbatchTensor, 'video_dino': videodinobatchTensor}}

    if 'object_type' in notnone_batches[0]:
        objecttypebatch = [b['object_type'] for b in notnone_batches]
        cond['y'].update({'object_type': objecttypebatch})

    if 'parents' in notnone_batches[0]:
        parentsbatch = [b['parents'] for b in notnone_batches]
        cond['y'].update({'parents': parentsbatch})

    if 'joints_names_embs' in notnone_batches[0]:
        jointsnamesembsbatch = [b['joints_names_embs'] for b in notnone_batches]
        jointsnamesembsbatchTensor = collate_tensors(jointsnamesembsbatch)
        cond['y'].update({'joints_names_embs': jointsnamesembsbatchTensor})

    if 'joints_relations' in notnone_batches[0]:
        jointsrelationsbatch = [b['joints_relations'] for b in notnone_batches]

    if 'graph_dist' in notnone_batches[0]:
        graphdistbatch = [b['graph_dist'] for b in notnone_batches]

    cond['y'].update({'joints_mask': jointsmaskbatchTensor})
    cond['y'].update({'n_joints': jointsnumbatchTensor})
    cond['y'].update({'joints_relations': torch.stack(jointsrelationsbatch)})
    cond['y'].update({'graph_dist': torch.stack(graphdistbatch)})

    return motion, cond

def truebones_batch_collate(batch):
    """Pad motion samples and collate them into model inputs and conditions."""
    max_joints = batch[0][-1]
    adapted_batch = []
    for b in batch:
        max_len, n_joints, n_feats = b[0].shape
        tpos_first_frame = torch.zeros((max_joints, n_feats))
        tpos_first_frame[:n_joints] = torch.tensor(b[3])
        motion = torch.zeros((max_len, max_joints, n_feats))
        motion[:, :b[0].shape[1], :] = torch.tensor(b[0])
        joints_names_embs = torch.zeros((max_joints, b[8].shape[1]))
        joints_names_embs[:n_joints] = torch.tensor(b[8])
        crop_start_ind = b[9]
        mean = torch.zeros((max_joints, n_feats))
        std = torch.ones((max_joints, n_feats))
        n_joints = b[0].shape[1]
        temporal_mask = b[5][:max_len + 1, :max_len + 1].clone()
        padded_joints_relations =  create_padded_relation(b[7], max_joints, n_joints)
        padded_graph_dist =  create_padded_relation(b[6], max_joints, n_joints)
        video_dino = torch.tensor(b[12])
        item = {
            'inp': motion.permute(1, 2, 0).float(),
            'n_joints': n_joints,
            'lengths': b[1],
            'parents': b[2],
            'temporal_mask' : temporal_mask,
            'graph_dist' : padded_graph_dist,
            'joints_relations':  padded_joints_relations,
            'joints_names_embs': joints_names_embs,
            'tpos_first_frame': tpos_first_frame,
            'crop_start_ind': crop_start_ind,
            'mean': mean,
            'std': std,
            'video_dino': video_dino,
        }
        adapted_batch.append(item)

    return truebones_collate(adapted_batch)
