"""Bound optional model context without altering authoritative execution data."""
import json
import re


def context_overflow(exc):
    text = str(exc).lower()
    return ('context length' in text or 'maximum context' in text or
            'input_tokens' in text or 'context budget' in text or
            'max_model_len' in text)


def compact_data(value, *, chars=8000, max_items=12, max_string=650):
    """Preserve complete JSON and indicate omissions explicitly.

    This is an advisory sample only; never use for execution or fingerprinting.
    """
    def sample(obj, limit, text_limit, depth=0):
        if depth > 9:
            return {'sample_omitted': 'nested context'}
        if isinstance(obj, dict):
            return {str(k):sample(v,limit,text_limit,depth+1) for k,v in obj.items()}
        if isinstance(obj,(list,tuple)):
            result = [sample(x,limit,text_limit,depth+1) for x in obj[:limit]]
            if len(obj)>limit:
                result.append({'sample_omitted_count':len(obj)-limit})
            return result
        if isinstance(obj,str) and len(obj)>text_limit:
            return obj[:text_limit]+' … [sample truncated]'
        return obj
    for limit, text_limit in [(max_items,max_string),(min(max_items,6),min(max_string,400)),(min(max_items,3),min(max_string,240)),(1,160)]:
        data=sample(value,limit,text_limit)
        if len(json.dumps(data,ensure_ascii=False,default=str))<=chars:
            return data
    # Do not cut JSON or pretend a partial list represents complete evidence.
    return {'sample_omitted':'context exceeds advisory budget','top_level_count':len(value) if isinstance(value,(list,dict)) else 1}


def token_limit_from_error(exc):
    match = re.search(r'maximum context length is\s+(\d+)',str(exc),re.I)
    return int(match.group(1)) if match else None
