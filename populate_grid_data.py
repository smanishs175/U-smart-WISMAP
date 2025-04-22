import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import async_session
from app.models.grid import Bus, Branch, Generator, Load, Substation, BalancingAuthority
import random
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample data
BA_NAMES = [
    ("California ISO", "CAISO"),
    ("Bonneville Power Administration", "BPA"),
    ("PacifiCorp", "PACW"),
    ("NV Energy", "NEVP"),
    ("Arizona Public Service", "AZPS"),
    ("Salt River Project", "SRP"),
    ("Public Service Company of New Mexico", "PNM"),
    ("El Paso Electric", "EPE"),
    ("Public Service Company of Colorado", "PSCO"),
    ("Western Area Power Administration", "WAPA")
]

GEN_TYPES = ["solar", "wind", "hydro", "natural_gas", "coal", "nuclear", "geothermal", "biomass"]

async def populate_grid_data():
    """
    Populate the database with sample grid data.
    """
    async with async_session() as db:
        # Check if we already have data
        from sqlalchemy.future import select
        result = await db.execute(select(BalancingAuthority))
        existing_bas = result.scalars().all()
        
        if existing_bas:
            logger.info("Grid data already exists. Skipping population.")
            return
        
        logger.info("Populating grid data...")
        
        # Create Balancing Authorities
        bas = []
        for name, abbr in BA_NAMES:
            ba = BalancingAuthority(
                name=name,
                abbreviation=abbr,
                geometry=f"POINT({random.uniform(-125, -100)} {random.uniform(30, 45)})",
                metadata_json={"region": "WECC"}
            )
            db.add(ba)
            bas.append(ba)
        
        await db.commit()
        logger.info(f"Created {len(bas)} balancing authorities")
        
        # Create Substations
        substations = []
        for i in range(50):
            ba = random.choice(bas)
            substation = Substation(
                name=f"Substation {i+1}",
                voltage=random.choice([115, 230, 345, 500]),
                ba_id=ba.id,
                geometry=f"POINT({random.uniform(-125, -100)} {random.uniform(30, 45)})",
                metadata_json={"owner": f"Utility {i%10 + 1}"}
            )
            db.add(substation)
            substations.append(substation)
        
        await db.commit()
        logger.info(f"Created {len(substations)} substations")
        
        # Create Buses
        buses = []
        for i in range(200):
            substation = random.choice(substations)
            bus = Bus(
                name=f"Bus {i+1}",
                base_kv=random.choice([115, 230, 345, 500]),
                substation_id=substation.id,
                ba_id=substation.ba_id,
                geometry=f"POINT({random.uniform(-125, -100)} {random.uniform(30, 45)})",
                metadata_json={"type": random.choice(["PQ", "PV", "Slack"])}
            )
            db.add(bus)
            buses.append(bus)
        
        await db.commit()
        logger.info(f"Created {len(buses)} buses")
        
        # Create Generators
        generators = []
        for i in range(100):
            bus = random.choice(buses)
            gen_type = random.choice(GEN_TYPES)
            p_max = random.uniform(50, 1000)
            generator = Generator(
                name=f"{gen_type.capitalize()} Generator {i+1}",
                bus_id=bus.id,
                gen_type=gen_type,
                p_max=p_max,
                p_min=p_max * 0.1,
                q_max=p_max * 0.3,
                q_min=-p_max * 0.3,
                status=random.choice([True, True, True, False]),  # 75% online
                ba_id=bus.ba_id,
                geometry=f"POINT({random.uniform(-125, -100)} {random.uniform(30, 45)})",
                metadata_json={"fuel_cost": random.uniform(10, 50)}
            )
            db.add(generator)
            generators.append(generator)
        
        await db.commit()
        logger.info(f"Created {len(generators)} generators")
        
        # Create Loads
        loads = []
        for i in range(150):
            bus = random.choice(buses)
            p_load = random.uniform(10, 500)
            load = Load(
                name=f"Load {i+1}",
                bus_id=bus.id,
                p_load=p_load,
                q_load=p_load * random.uniform(0.1, 0.3),
                status=random.choice([True, True, True, False]),  # 75% online
                ba_id=bus.ba_id,
                geometry=f"POINT({random.uniform(-125, -100)} {random.uniform(30, 45)})",
                metadata_json={"type": random.choice(["residential", "commercial", "industrial"])}
            )
            db.add(load)
            loads.append(load)
        
        await db.commit()
        logger.info(f"Created {len(loads)} loads")
        
        # Create Branches
        branches = []
        for i in range(250):
            from_bus = random.choice(buses)
            to_bus = random.choice([b for b in buses if b.id != from_bus.id])
            branch = Branch(
                name=f"Branch {i+1}",
                from_bus_id=from_bus.id,
                to_bus_id=to_bus.id,
                r=random.uniform(0.001, 0.05),
                x=random.uniform(0.01, 0.2),
                b=random.uniform(0.0001, 0.01),
                rate_a=random.uniform(100, 1000),
                rate_b=random.uniform(110, 1100),
                rate_c=random.uniform(120, 1200),
                status=random.choice([True, True, True, False]),  # 75% online
                ba_id=from_bus.ba_id,
                geometry=f"LINESTRING({random.uniform(-125, -100)} {random.uniform(30, 45)}, {random.uniform(-125, -100)} {random.uniform(30, 45)})",
                metadata_json={"type": random.choice(["line", "transformer"])}
            )
            db.add(branch)
            branches.append(branch)
        
        await db.commit()
        logger.info(f"Created {len(branches)} branches")
        
        # Create Energy Emergency Alerts
        eea_events = []
        start_date = datetime(2020, 7, 1).date()
        end_date = datetime(2020, 7, 31).date()
        
        for i in range(20):
            ba = random.choice(bas)
            level = random.choice([1, 1, 2, 2, 3])  # More level 1 and 2 than 3
            date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
            
            description = ""
            if level == 1:
                description = f"Conservation alert issued for {ba.name} service area"
            elif level == 2:
                description = f"Load management procedures in effect for {ba.name}"
            else:
                description = f"Firm load interruptions imminent or in progress for {ba.name}"
            
            eea = EnergyEmergencyAlert(
                date=date,
                level=level,
                description=description,
                ba_id=ba.id,
                metadata_json={
                    "affected_mw": random.uniform(100, 5000),
                    "duration_hours": random.uniform(1, 8)
                }
            )
            db.add(eea)
            eea_events.append(eea)
        
        await db.commit()
        logger.info(f"Created {len(eea_events)} EEA events")
        
        logger.info("Grid data population complete!")

if __name__ == "__main__":
    asyncio.run(populate_grid_data())
