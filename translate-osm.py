#!/usr/bin/env python3
import os
import csv
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional
from openai import OpenAI

# pip install tiktoken
from typing import List, Dict, Any, Tuple
import tiktoken

# --- Pricing per 1,000 tokens (USD). Update if pricing changes. ---
MODEL_PRICING = {
    "gpt-5.1":     {"input": 1.25/1000, "output": 10.00/1000},
    "gpt-5-mini":  {"input": 0.25/1000, "output": 2.00/1000},
    "gpt-5-nano":  {"input": 0.05/1000, "output": 0.40/1000},
    "gpt-4.1":     {"input": 3.00/1000, "output": 12.00/1000},
    "gpt-4.1-mini":{"input": 0.80/1000, "output": 3.20/1000},
    "gpt-4.1-nano":{"input": 0.20/1000, "output": 0.80/1000},
    "gpt-4o-mini": {"input": 4.0/1000, "output": 16.00/1000},
}

# --- Per-message token overhead heuristics for chat formatting ---
# These are based on OpenAI/tiktoken examples for chat models.
# If a model isn't listed, we fall back to a sensible default.
CHAT_OVERHEAD = {
    # name: (tokens_per_message, tokens_per_name, priming_tokens)
    "gpt-5.1":       (3, 1, 3),
    "gpt-5-mini":    (3, 1, 3),
    "gpt-5-nano":    (3, 1, 3),
    "gpt-4.1":       (3, 1, 3),
    "gpt-4.1-mini":  (3, 1, 3),
    "gpt-4.1-nano":  (3, 1, 3),
    "gpt-4o-mini":   (3, 1, 3),
}

def _encoding_for(model: str):
    """Get a tiktoken encoding for the given model, with a safe fallback."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Safe fallback for most modern OpenAI chat models
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Final fallback if tiktoken is very old
            return tiktoken.get_encoding(tiktoken.list_encoding_names()[0])

def count_tokens(text: str, model: str) -> int:
    """Count tokens for a single string."""
    enc = _encoding_for(model)
    return len(enc.encode(text))

def count_message_tokens(
    messages: List[Dict[str, Any]],
    model: str = "gpt-4o-mini"
) -> int:
    """
    Count tokens for a chat conversation.
    messages: list like [{"role":"user","content":"Hi"}, {"role":"assistant","content":"Hello"}]
    """
    enc = _encoding_for(model)
    tpm, tpn, priming = CHAT_OVERHEAD.get(model, (3, 1, 3))

    total = priming  # e.g., <|start|> system priming, etc.

    for m in messages:
        total += tpm
        # role isn't usually counted through enc.encode in older recipes; it's part of overhead
        # content + name are encoded normally:
        content = m.get("content", "")
        total += len(enc.encode(content))
        name = m.get("name")
        if name:
            total += tpn
    # Assistant reply typically gets an extra 3 tokens in older templates; we skip here,
    # and instead let callers pass expected output tokens separately.
    return total

def estimate_chat_cost(
    messages: List[Dict[str, Any]],
    model: str,
    expected_output_tokens: int = 0
) -> Tuple[int, float]:
    """
    Return (input_tokens, estimated_total_cost_usd)
    """
    if model not in MODEL_PRICING:
        raise ValueError(f"Unknown model '{model}'. Please add it to MODEL_PRICING.")

    input_tokens = count_message_tokens(messages, model)
    pricing = MODEL_PRICING[model]

    input_cost  = (input_tokens / 1000) * pricing["input"]
    output_cost = (expected_output_tokens / 1000) * pricing["output"]
    return input_tokens, round(input_cost + output_cost, 6)

def translateText(target_language: str,
                  payload: list[str],
                  openAiModel: str):
    system_msg = (
        "You are an expert in Open Street Map and you are creating short text descriptions"
        f"in ${target_language} for various tags and tag values that Open Street Map uses.\n\n"
        "Task: For each string in the list, which we call $input, create a very short string which we call $output."
        "$output is to be used in an app to describe the mapped feature to the users as text. It should be the"
        "absolutely shortest description possible as it is read out using text to speech. The words in $output"
        "should be capitalized. Return ONLY valid JSON with this exact shape:\n"
        "{ 'results': [\"<!-- Open Street Map term $input. -->\n"
        "<string name=\"osm_$input\" tools:ignore=\"MissingTranslation\">$output</string>\n,...]}\n"
        "Do not add explanations or extra keys.\n\n"
    )

    user_msg = (
        "Here is the list of strings to translate as JSON:\n" +
        json.dumps(payload, ensure_ascii=False)
    )

    conversation = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # Estimate the cost of translation
    in_tokens = count_message_tokens(conversation, openAiModel)
    print("Input tokens:", in_tokens)

    # The number of output tokens is likely to be similar to the number of input tokens
    conversation_output = [
        {"role": "user", "content": user_msg},
    ]
    expected_out = count_message_tokens(conversation_output, openAiModel)
    in_tokens, cost = estimate_chat_cost(conversation, openAiModel, expected_out)
    print(f"Estimated cost (incl. ~{expected_out} output tokens): ${cost}")

    if True:
        openAiClient = OpenAI(
            api_key=""
        )
        resp = openAiClient.chat.completions.create(
            model=openAiModel,
            messages=conversation,
            response_format={"type": "json_object"}    # Ask the model to return strict JSON (no prose)
        )

        content = resp.choices[0].message.content
        data = json.loads(content)

        output_tokens = count_tokens(content, openAiModel)
        print("Actual output tokens:", output_tokens)

        return data["results"]
    

def write_file(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)
        
def main():
    p = argparse.ArgumentParser(description="Describe OSM tags")
    args = p.parse_args()

    language_code = "en"
    language_string = "US English"

    input_strings = [
        "crossing","construction","dangerous_area","townhall","steps","elevator","walking_path","pedestrian_street",
        "bicycle_path","residential_street","service_road","road","highway","highway_named","highway_refed","intersection",
        "roundabout","highway_ramp","merging_lane","office","school","roof","convenience","entrance","assembly_point","cycle_barrier",
        "turnstile","cattle_grid","gate","lift_gate","toilets","parking","parking_entrance","bench","taxi","post_office","post_box",
        "waste_basket","shower","bicycle_parking","cafe","restaurant","telephone","fuel","bank","atm","atm_named","atm_refed",
        "bus_stop","recycling","fountain","place_of_worship","drinking_water","car_wash","vending_machine","playground","pitch",
        "swimming_pool","garden","park","picnic_table","picnic_site","information","fire_extinguisher","defibrillator","guide",
        "water","fire_hose","fire_flapper","information_point","wetland","mud","access_point","life_ring","generic_info","turntable",
        "survey_point","snow_net","silo","mast","bird_hide","transformer_tower","generic_object","signal","rock","kiln","crane",
        "rune_stone","milestone","lifeguard_platform","water_tank","sty","navigationaid","terminal","water_tap","water_well",
        "petroleum_well","cross","gallows","speed_camera","siren","pylon","mineshaft","flagpole","optical_telegraph","cannon",
        "boundary_stone","shed","traffic_cones","firepit","stone","surveillance","monitoring_station","wayside_shrine","wayside_cross",
        "tomb","traffic_signals","hut","static_caravan","bollard","block","waste_disposal","photo_booth","bbq","shop","newsagent",
        "anime","musical_instrument","vacuum_cleaner","mobile_phone","carpet","trade","garden_centre","florist","fireplace","massage",
        "herbalist","bag","pastry","deli","beverages","alcohol","substation","travel_agent","research","newspaper","ammunition",
        "wildlife_hide","watchmaker","tinsmith","sun_protection","sculptor","metal_construction","handicraft","cowshed","cabin","barn",
        "warehouse","houseboat","book_store","generic_place","hunting_stand","game_feeding","crypt","animal_shelter","animal_boarding",
        "blood_donation","nursing_home","dentist","baby_hatch","language_school","public_bookcase","biergarten","running","glaziery",
        "garages","retail","hotel","camp_site","rugby_league","roller_skating","multi","ice_hockey","hapkido","croquet","cricket",
        "cockfighting","boxing","bmx","billiards","toys","pyrotechnics","laundry","funeral_directors","dry_cleaning","copyshop",
        "chalet","apartment","water_ski","water_polo","table_soccer","table_tennis","skateboard","sailing","safety_training","rowing",
        "model_aerodrome","korfball","ice_stock","gymnastics","football","field_hockey","equestrian","cycling","curling","cricket_nets",
        "cliff_diving","boules","bobsleigh","baseball","aikido","10pin","weapons","pet","money_lender","gift","books","bookmaker",
        "photo","craft","motorcycle","hunting","window_blind","curtain","antiques","paint","tattoo","nutrition_supplements",
        "hearing_aids","cosmetics","watches","jewelry","boutique","baby_goods","tea","pasta","coffee","quango","political_party",
        "association","architect","advertising_agency","summer_camp","dance","amusement_arcade","adult_gaming_centre",
        "window_construction","upholsterer","shoemaker","sawmill","pottery","key_cutter","hvac","clockmaker","carpenter","builder",
        "bookbinder","boatbuilder","brewery","blacksmith","basket_maker","greenhouse","farm_auxiliary","civic","bungalow","detached",
        "hair_dresser","clothing_store","user","dojo","nightclub","community_centre","brothel","veterinary","social_facility","clinic",
        "charging_station","kindergarten","ice_cream","fast_food","commercial","canoe","scuba_diving","fishing","optician","confectionery",
        "bunker","sleeping_pods","motel","guest_house","wrestling","toboggan","skiing","rc_car","paddle_tennis","hockey","fencing","bowls",
        "badminton","archery","american_football","travel_agency","tobacco","e_cigarette","video","car_repair","hifi","lamps","kitchen",
        "interior_decoration","houseware","erotic","beauty","wine","dairy","cheese","bakery","telecommunication","tax","real_estate_agent",
        "notary","ngo","lawyer","it","foundation","employment_agency","educational_institution","adoption_agency","miniature_golf",
        "building","winery","tiler","chimney_sweeper","stand_builder","saddler","plumber","plasterer","painter","jeweller","floorer",
        "distillery","carpet_layer","beekeeper","public","dormitory","apartments","internet_cafe","shoe_shop","generic_shop","coffee_shop",
        "coworking_space","stripclub","ev_charging","pub","obstacle_course","volleyball","tennis","soccer","shooting","rugby_union",
        "orienteering","netball","motor","kitesurfing","karting","judo","horseshoes","handball","golf","gaelic_games","diving","darts",
        "climbing_adventure","basketball","bandy","australian_football","9pin","vacant","lottery","trophy","music","games","tyres",
        "sports","outdoor","car","electronics","computer","furniture","candles","hardware","gas","energy","doityourself",
        "bathroom_furnishing","medical_supply","variety_store","second_hand","charity","fashion","fabric","clothes","butcher",
        "water_utility","realtor","company","accountant","bunker_silo","hackerspace","lifeguard_base","roofer","rigger","parquet_layer",
        "gardener","stable","garage","transportation","house","helipad","apron","consumer_electronics_store","speciality_store","defined",
        "sauna","gym","crematorium","gambling","music_school","bar","farm","bicycle","tailor","locksmith","industrial","wilderness_hut",
        "hostel","caravan_site","weightlifting","taekwondo","swimming","surfing","skating","racquet","pelota","paragliding","parachuting",
        "motocross","ice_skating","horse_racing","dog_racing","climbing","chess","canadian_football","beachvolleyball","base","athletics",
        "pawnbroker","ticket","stationery","video_games","model","frame","art","car_parts","radiotechnics","bed","garden_furniture",
        "electrical","perfumery","hairdresser","drugstore","shoes","leather","general","seafood","organic","greengrocer","chocolate",
        "brewing_supplies","tax_advisor","private_investigator","government","forestry","estate_agent","spring","golf_course","ses_station",
        "lifeguard_place","stonemason","scaffolder","sailmaker","photographic_laboratory","photographer","insulation","electrician",
        "dressmaker","caterer","terrace","toy_shop","dive_centre","swingerclub","doctors","driving_school","free_flying","religion","kiosk",
        "residential","food","waterfall","boatyard","theme_park","roundhouse","generator","beach","naval_base","works","water_works",
        "telescope","pier","observatory","reservoir","monument","battlefield","planetarium","social_centre","prison","courthouse","bridge",
        "hangar","tower","attraction","zoo","gallery","artwork","alpine_hut","plant","insurance","airfield","water_tower","pumping_station",
        "hot_water_tank","campanile","sports_centre","fitness_centre","beach_resort","village_green","ship","memorial","synagogue",
        "mosque","chapel","cathedral","train_terminal","college","arts_centre","ranger_station","hospital","track","conference_centre",
        "viewpoint","supermarket","peak","storage_tank","lighthouse","beacon","port","archaeological_site","train_station","shrine","church",
        "historic_monument","generic_landmark","tourism_museum","register_office","grave_yard","marketplace","fire_station","ruins","weir",
        "museum","mall","volcano","hot_spring","glacier","wastewater_plant","offshore_platform","gasometer","water_park","bandstand","wreck",
        "pillory","monastery","locomotive","fort","services","lifeguard_tower","temple","national_park","heliport","public_park",
        "department_store","studio","public_building","clock","casino","ferry_terminal","stadium","dam","dock","geyser","bay","barracks",
        "windmill","watermill","communications_tower","swimming_area","slipway","nature_reserve","marina","ice_rink","manor","city_gate",
        "castle","aircraft","digester","sally_port","aerodrome","shopping_mall","cinema","rescue_station","airport","theatre","library",
        "university","police","embassy","bus_station","station","toll_booth","lift","unmanaged_crossing","pharmacy","kneipp_water_cure",
        "food_court","chemist","checkpoint","dog_park","kissing_gate","car_rental","pedestrianised_area","escalator","shelter","water_point",
        "subway_entrance","cave_entrance","swing_gate","stile","car_sharing","customer_service","watering_place","platform","horse_stile",
        "bureau_de_change","stairs","bicycle_rental","tram_stop","subway","hampshire_gate","full-height_turnstile","boat_sharing",
        "help_point","open_space","spending_area","bicycle_repair_station","motorcycle_barrier","kent_carriage_gap","shared_space","cliff",
        "training_area","log","jersey_barrier","construction_site","ridge","nuclear_explosion_site","dyke","sump_buster","rope","debris",
        "road_works","lock_gate","sinkhole","range","ambulance_station","spikes","generic_hazard","contact_line","danger_area","chain",
        "parking_space","motorcycle_parking"
    ]

    openAiModel="gpt-5-mini"
    target_language = language_string
    out_path = Path(f"{target_language}-{openAiModel}-osm.xml")

    strings_at_a_time = 100
    start = 0
    end = strings_at_a_time
    accumulated_results = ""
    timeout_count = 0
    while start < end:
        try:
            results = translateText(target_language, input_strings[start:end], openAiModel)
            if results != None:
                # Transform results into simple JSON format that can be accepted by Weblate
                for result in results:
                    accumulated_results += result
                    accumulated_results += "\n"

                start = start + len(results)
            else:
                start = start + strings_at_a_time

            end = start + strings_at_a_time
            if end > len(input_strings):
                end = len(input_strings)

        except Exception as inst:
            print(type(inst))    # the exception type
            print(inst.args)     # arguments stored in .args
            print(inst)

            timeout_count = timeout_count + 1
            time.sleep(10)
            # Try a smaller number of strings
            strings_at_a_time = strings_at_a_time = 10
            end = start + strings_at_a_time
        
        if timeout_count > 5:
            break

    write_file(out_path, accumulated_results)

if __name__ == "__main__":
    main()
