import asyncio

async def count():
    print("one")
    await asyncio.sleep(1)
    print("two")

async def main():
    await asyncio.gather(count(), count(), count())

if __name__ == "__main__":
    asyncio.run(main())